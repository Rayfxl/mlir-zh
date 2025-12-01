# 'rocdl' Dialect

- [操作](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#operations)
  - [`rocdl.ballot`(ROCDL::BallotOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlballot-rocdlballotop)
  - [`rocdl.barrier`(ROCDL::BarrierOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlbarrier-rocdlbarrierop)
  - [`rocdl.cluster.id.x`(ROCDL::ClusterIdXOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlclusteridx-rocdlclusteridxop)
  - [`rocdl.cluster.id.y`(ROCDL::ClusterIdYOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlclusteridy-rocdlclusteridyop)
  - [`rocdl.cluster.id.z`(ROCDL::ClusterIdZOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlclusteridz-rocdlclusteridzop)
  - [`rocdl.cluster.load.async.to.lds.b128`(ROCDL::ClusterLoadAsyncToLDSB128Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlclusterloadasynctoldsb128-rocdlclusterloadasynctoldsb128op)
  - [`rocdl.cluster.load.async.to.lds.b32`(ROCDL::ClusterLoadAsyncToLDSB32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlclusterloadasynctoldsb32-rocdlclusterloadasynctoldsb32op)
  - [`rocdl.cluster.load.async.to.lds.b64`(ROCDL::ClusterLoadAsyncToLDSB64Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlclusterloadasynctoldsb64-rocdlclusterloadasynctoldsb64op)
  - [`rocdl.cluster.load.async.to.lds.b8`(ROCDL::ClusterLoadAsyncToLDSB8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlclusterloadasynctoldsb8-rocdlclusterloadasynctoldsb8op)
  - [`rocdl.cvt.f32.bf8`(ROCDL::CvtF32Bf8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtf32bf8-rocdlcvtf32bf8op)
  - [`rocdl.cvt.f32.fp8`(ROCDL::CvtF32Fp8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtf32fp8-rocdlcvtf32fp8op)
  - [`rocdl.cvt.pk.bf8.f32`(ROCDL::CvtPkBf8F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtpkbf8f32-rocdlcvtpkbf8f32op)
  - [`rocdl.cvt.pk.f32.bf8`(ROCDL::CvtPkF32Bf8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtpkf32bf8-rocdlcvtpkf32bf8op)
  - [`rocdl.cvt.pk.f32.fp8`(ROCDL::CvtPkF32Fp8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtpkf32fp8-rocdlcvtpkf32fp8op)
  - [`rocdl.cvt.pk.fp8.f32`(ROCDL::CvtPkFp8F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtpkfp8f32-rocdlcvtpkfp8f32op)
  - [`rocdl.cvt.pkrtz`(ROCDL::CvtPkRtz)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtpkrtz-rocdlcvtpkrtz)
  - [`rocdl.cvt.scale.pk16.bf16.bf6`(ROCDL::CvtPkScalePk16Bf16Bf6Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalepk16bf16bf6-rocdlcvtpkscalepk16bf16bf6op)
  - [`rocdl.cvt.scale.pk16.bf16.fp6`(ROCDL::CvtPkScalePk16Bf16Fp6Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalepk16bf16fp6-rocdlcvtpkscalepk16bf16fp6op)
  - [`rocdl.cvt.scale.pk16.f16.bf6`(ROCDL::CvtPkScalePk16F16Bf6Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalepk16f16bf6-rocdlcvtpkscalepk16f16bf6op)
  - [`rocdl.cvt.scale.pk16.f16.fp6`(ROCDL::CvtPkScalePk16F16Fp6Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalepk16f16fp6-rocdlcvtpkscalepk16f16fp6op)
  - [`rocdl.cvt.scale.pk16.f32.bf6`(ROCDL::CvtPkScalePk16F32Bf6Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalepk16f32bf6-rocdlcvtpkscalepk16f32bf6op)
  - [`rocdl.cvt.scale.pk16.f32.fp6`(ROCDL::CvtPkScalePk16F32Fp6Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalepk16f32fp6-rocdlcvtpkscalepk16f32fp6op)
  - [`rocdl.cvt.scale.pk8.bf16.bf8`(ROCDL::CvtPkScalePk8Bf16Bf8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalepk8bf16bf8-rocdlcvtpkscalepk8bf16bf8op)
  - [`rocdl.cvt.scale.pk8.bf16.fp4`(ROCDL::CvtPkScalePk8Bf16Fp4Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalepk8bf16fp4-rocdlcvtpkscalepk8bf16fp4op)
  - [`rocdl.cvt.scale.pk8.bf16.fp8`(ROCDL::CvtPkScalePk8Bf16Fp8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalepk8bf16fp8-rocdlcvtpkscalepk8bf16fp8op)
  - [`rocdl.cvt.scale.pk8.f16.bf8`(ROCDL::CvtPkScalePk8F16Bf8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalepk8f16bf8-rocdlcvtpkscalepk8f16bf8op)
  - [`rocdl.cvt.scale.pk8.f16.fp4`(ROCDL::CvtPkScalePk8F16Fp4Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalepk8f16fp4-rocdlcvtpkscalepk8f16fp4op)
  - [`rocdl.cvt.scale.pk8.f16.fp8`(ROCDL::CvtPkScalePk8F16Fp8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalepk8f16fp8-rocdlcvtpkscalepk8f16fp8op)
  - [`rocdl.cvt.scale.pk8.f32.bf8`(ROCDL::CvtPkScalePk8F32Bf8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalepk8f32bf8-rocdlcvtpkscalepk8f32bf8op)
  - [`rocdl.cvt.scale.pk8.f32.fp4`(ROCDL::CvtPkScalePk8F32Fp4Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalepk8f32fp4-rocdlcvtpkscalepk8f32fp4op)
  - [`rocdl.cvt.scale.pk8.f32.fp8`(ROCDL::CvtPkScalePk8F32Fp8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalepk8f32fp8-rocdlcvtpkscalepk8f32fp8op)
  - [`rocdl.cvt.scalef32.2xpk16.bf6.f32`(ROCDL::CvtScaleF322xPk16Bf6F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef322xpk16bf6f32-rocdlcvtscalef322xpk16bf6f32op)
  - [`rocdl.cvt.scalef32.2xpk16.fp6.f32`(ROCDL::CvtScaleF322xPk16Fp6F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef322xpk16fp6f32-rocdlcvtscalef322xpk16fp6f32op)
  - [`rocdl.cvt.scalef32.f16.bf8`(ROCDL::CvtScaleF32F16Bf8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32f16bf8-rocdlcvtscalef32f16bf8op)
  - [`rocdl.cvt.scalef32.f16.fp8`(ROCDL::CvtScaleF32F16Fp8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32f16fp8-rocdlcvtscalef32f16fp8op)
  - [`rocdl.cvt.scalef32.f32.bf8`(ROCDL::CvtScaleF32F32Bf8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32f32bf8-rocdlcvtscalef32f32bf8op)
  - [`rocdl.cvt.scalef32.f32.fp8`(ROCDL::CvtScaleF32F32Fp8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32f32fp8-rocdlcvtscalef32f32fp8op)
  - [`rocdl.cvt.scalef32.pk.bf16.bf8`(ROCDL::CvtScaleF32PkBf16Bf8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkbf16bf8-rocdlcvtscalef32pkbf16bf8op)
  - [`rocdl.cvt.scalef32.pk.bf16.fp4`(ROCDL::CvtScaleF32PkBf16Fp4Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkbf16fp4-rocdlcvtscalef32pkbf16fp4op)
  - [`rocdl.cvt.scalef32.pk.bf16.fp8`(ROCDL::CvtScaleF32PkBf16Fp8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkbf16fp8-rocdlcvtscalef32pkbf16fp8op)
  - [`rocdl.cvt.scalef32.pk.bf8.bf16`(ROCDL::CvtScaleF32PkBf8Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkbf8bf16-rocdlcvtscalef32pkbf8bf16op)
  - [`rocdl.cvt.scalef32.pk.bf8.f16`(ROCDL::CvtScaleF32PkBf8F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkbf8f16-rocdlcvtscalef32pkbf8f16op)
  - [`rocdl.cvt.scalef32.pk.bf8.f32`(ROCDL::CvtScaleF32PkBf8F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkbf8f32-rocdlcvtscalef32pkbf8f32op)
  - [`rocdl.cvt.scalef32.pk.f16.bf8`(ROCDL::CvtScaleF32PkF16Bf8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkf16bf8-rocdlcvtscalef32pkf16bf8op)
  - [`rocdl.cvt.scalef32.pk.f16.fp4`(ROCDL::CvtScaleF32PkF16Fp4Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkf16fp4-rocdlcvtscalef32pkf16fp4op)
  - [`rocdl.cvt.scalef32.pk.f16.fp8`(ROCDL::CvtScaleF32PkF16Fp8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkf16fp8-rocdlcvtscalef32pkf16fp8op)
  - [`rocdl.cvt.scalef32.pk.f32.bf8`(ROCDL::CvtScaleF32PkF32Bf8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkf32bf8-rocdlcvtscalef32pkf32bf8op)
  - [`rocdl.cvt.scalef32.pk.f32.fp4`(ROCDL::CvtScaleF32PkF32Fp4Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkf32fp4-rocdlcvtscalef32pkf32fp4op)
  - [`rocdl.cvt.scalef32.pk.f32.fp8`(ROCDL::CvtScaleF32PkF32Fp8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkf32fp8-rocdlcvtscalef32pkf32fp8op)
  - [`rocdl.cvt.scalef32.pk.fp4.bf16`(ROCDL::CvtScaleF32PkFp4Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkfp4bf16-rocdlcvtscalef32pkfp4bf16op)
  - [`rocdl.cvt.scalef32.pk.fp4.f16`(ROCDL::CvtScaleF32PkFp4F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkfp4f16-rocdlcvtscalef32pkfp4f16op)
  - [`rocdl.cvt.scalef32.pk.fp4.f32`(ROCDL::CvtScaleF32PkFp4F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkfp4f32-rocdlcvtscalef32pkfp4f32op)
  - [`rocdl.cvt.scalef32.pk.fp8.bf16`(ROCDL::CvtScaleF32PkFp8Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkfp8bf16-rocdlcvtscalef32pkfp8bf16op)
  - [`rocdl.cvt.scalef32.pk.fp8.f16`(ROCDL::CvtScaleF32PkFp8F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkfp8f16-rocdlcvtscalef32pkfp8f16op)
  - [`rocdl.cvt.scalef32.pk.fp8.f32`(ROCDL::CvtScaleF32PkFp8F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pkfp8f32-rocdlcvtscalef32pkfp8f32op)
  - [`rocdl.cvt.scalef32.pk16.bf6.bf16`(ROCDL::CvtScaleF32Pk16Bf6Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk16bf6bf16-rocdlcvtscalef32pk16bf6bf16op)
  - [`rocdl.cvt.scalef32.pk16.bf6.f16`(ROCDL::CvtScaleF32Pk16Bf6F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk16bf6f16-rocdlcvtscalef32pk16bf6f16op)
  - [`rocdl.cvt.scalef32.pk16.bf6.f32`(ROCDL::CvtScaleF32Pk16Bf6F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk16bf6f32-rocdlcvtscalef32pk16bf6f32op)
  - [`rocdl.cvt.scalef32.pk16.fp6.bf16`(ROCDL::CvtScaleF32Pk16Fp6Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk16fp6bf16-rocdlcvtscalef32pk16fp6bf16op)
  - [`rocdl.cvt.scalef32.pk16.fp6.f16`(ROCDL::CvtScaleF32Pk16Fp6F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk16fp6f16-rocdlcvtscalef32pk16fp6f16op)
  - [`rocdl.cvt.scalef32.pk16.fp6.f32`(ROCDL::CvtScaleF32Pk16Fp6F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk16fp6f32-rocdlcvtscalef32pk16fp6f32op)
  - [`rocdl.cvt.scalef32.pk32.bf16.bf6`(ROCDL::CvtScaleF32Pk32Bf16Bf6Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk32bf16bf6-rocdlcvtscalef32pk32bf16bf6op)
  - [`rocdl.cvt.scalef32.pk32.bf16.fp6`(ROCDL::CvtScaleF32Pk32Bf16Fp6Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk32bf16fp6-rocdlcvtscalef32pk32bf16fp6op)
  - [`rocdl.cvt.scalef32.pk32.bf6.bf16`(ROCDL::CvtScaleF32Pk32Bf6Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk32bf6bf16-rocdlcvtscalef32pk32bf6bf16op)
  - [`rocdl.cvt.scalef32.pk32.bf6.f16`(ROCDL::CvtScaleF32Pk32Bf6F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk32bf6f16-rocdlcvtscalef32pk32bf6f16op)
  - [`rocdl.cvt.scalef32.pk32.f16.bf6`(ROCDL::CvtScaleF32Pk32F16Bf6Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk32f16bf6-rocdlcvtscalef32pk32f16bf6op)
  - [`rocdl.cvt.scalef32.pk32.f16.fp6`(ROCDL::CvtScaleF32Pk32F16Fp6Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk32f16fp6-rocdlcvtscalef32pk32f16fp6op)
  - [`rocdl.cvt.scalef32.pk32.f32.bf6`(ROCDL::CvtScaleF32Pk32F32Bf6Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk32f32bf6-rocdlcvtscalef32pk32f32bf6op)
  - [`rocdl.cvt.scalef32.pk32.f32.fp6`(ROCDL::CvtScaleF32Pk32F32Fp6Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk32f32fp6-rocdlcvtscalef32pk32f32fp6op)
  - [`rocdl.cvt.scalef32.pk32.fp6.bf16`(ROCDL::CvtScaleF32Pk32Fp6Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk32fp6bf16-rocdlcvtscalef32pk32fp6bf16op)
  - [`rocdl.cvt.scalef32.pk32.fp6.f16`(ROCDL::CvtScaleF32Pk32Fp6F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk32fp6f16-rocdlcvtscalef32pk32fp6f16op)
  - [`rocdl.cvt.scalef32.pk8.bf8.bf16`(ROCDL::CvtScaleF32Pk8Bf8Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk8bf8bf16-rocdlcvtscalef32pk8bf8bf16op)
  - [`rocdl.cvt.scalef32.pk8.bf8.f16`(ROCDL::CvtScaleF32Pk8Bf8F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk8bf8f16-rocdlcvtscalef32pk8bf8f16op)
  - [`rocdl.cvt.scalef32.pk8.bf8.f32`(ROCDL::CvtScaleF32Pk8Bf8F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk8bf8f32-rocdlcvtscalef32pk8bf8f32op)
  - [`rocdl.cvt.scalef32.pk8.fp4.bf16`(ROCDL::CvtScaleF32Pk8Fp4Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk8fp4bf16-rocdlcvtscalef32pk8fp4bf16op)
  - [`rocdl.cvt.scalef32.pk8.fp4.f16`(ROCDL::CvtScaleF32Pk8Fp4F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk8fp4f16-rocdlcvtscalef32pk8fp4f16op)
  - [`rocdl.cvt.scalef32.pk8.fp4.f32`(ROCDL::CvtScaleF32Pk8Fp4F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk8fp4f32-rocdlcvtscalef32pk8fp4f32op)
  - [`rocdl.cvt.scalef32.pk8.fp8.bf16`(ROCDL::CvtScaleF32Pk8Fp8Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk8fp8bf16-rocdlcvtscalef32pk8fp8bf16op)
  - [`rocdl.cvt.scalef32.pk8.fp8.f16`(ROCDL::CvtScaleF32Pk8Fp8F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk8fp8f16-rocdlcvtscalef32pk8fp8f16op)
  - [`rocdl.cvt.scalef32.pk8.fp8.f32`(ROCDL::CvtScaleF32Pk8Fp8F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32pk8fp8f32-rocdlcvtscalef32pk8fp8f32op)
  - [`rocdl.cvt.scalef32.sr.bf8.bf16`(ROCDL::CvtScaleF32SrBf8BF16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srbf8bf16-rocdlcvtscalef32srbf8bf16op)
  - [`rocdl.cvt.scalef32.sr.bf8.f16`(ROCDL::CvtScaleF32SrBf8F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srbf8f16-rocdlcvtscalef32srbf8f16op)
  - [`rocdl.cvt.scalef32.sr.bf8.f32`(ROCDL::CvtScaleF32SrBf8F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srbf8f32-rocdlcvtscalef32srbf8f32op)
  - [`rocdl.cvt.scalef32.sr.fp8.bf16`(ROCDL::CvtScaleF32SrFp8BF16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srfp8bf16-rocdlcvtscalef32srfp8bf16op)
  - [`rocdl.cvt.scalef32.sr.fp8.f16`(ROCDL::CvtScaleF32SrFp8F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srfp8f16-rocdlcvtscalef32srfp8f16op)
  - [`rocdl.cvt.scalef32.sr.fp8.f32`(ROCDL::CvtScaleF32SrFp8F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srfp8f32-rocdlcvtscalef32srfp8f32op)
  - [`rocdl.cvt.scalef32.sr.pk.fp4.bf16`(ROCDL::CvtScaleF32SrPkFp4Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpkfp4bf16-rocdlcvtscalef32srpkfp4bf16op)
  - [`rocdl.cvt.scalef32.sr.pk.fp4.f16`(ROCDL::CvtScaleF32SrPkFp4F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpkfp4f16-rocdlcvtscalef32srpkfp4f16op)
  - [`rocdl.cvt.scalef32.sr.pk.fp4.f32`(ROCDL::CvtScaleF32SrPkFp4F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpkfp4f32-rocdlcvtscalef32srpkfp4f32op)
  - [`rocdl.cvt.scalef32.sr.pk16.bf6.bf16`(ROCDL::CvtScaleF32SrPk16Bf6Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk16bf6bf16-rocdlcvtscalef32srpk16bf6bf16op)
  - [`rocdl.cvt.scalef32.sr.pk16.bf6.f16`(ROCDL::CvtScaleF32SrPk16Bf6F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk16bf6f16-rocdlcvtscalef32srpk16bf6f16op)
  - [`rocdl.cvt.scalef32.sr.pk16.bf6.f32`(ROCDL::CvtScaleF32SrPk16Bf6F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk16bf6f32-rocdlcvtscalef32srpk16bf6f32op)
  - [`rocdl.cvt.scalef32.sr.pk16.fp6.bf16`(ROCDL::CvtScaleF32SrPk16Fp6Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk16fp6bf16-rocdlcvtscalef32srpk16fp6bf16op)
  - [`rocdl.cvt.scalef32.sr.pk16.fp6.f16`(ROCDL::CvtScaleF32SrPk16Fp6F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk16fp6f16-rocdlcvtscalef32srpk16fp6f16op)
  - [`rocdl.cvt.scalef32.sr.pk16.fp6.f32`(ROCDL::CvtScaleF32SrPk16Fp6F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk16fp6f32-rocdlcvtscalef32srpk16fp6f32op)
  - [`rocdl.cvt.scalef32.sr.pk32.bf6.bf16`(ROCDL::CvtScaleF32SrPk32Bf6Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk32bf6bf16-rocdlcvtscalef32srpk32bf6bf16op)
  - [`rocdl.cvt.scalef32.sr.pk32.bf6.f16`(ROCDL::CvtScaleF32SrPk32Bf6F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk32bf6f16-rocdlcvtscalef32srpk32bf6f16op)
  - [`rocdl.cvt.scalef32.sr.pk32.bf6.f32`(ROCDL::CvtScaleF32SrPk32Bf6F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk32bf6f32-rocdlcvtscalef32srpk32bf6f32op)
  - [`rocdl.cvt.scalef32.sr.pk32.fp6.bf16`(ROCDL::CvtScaleF32SrPk32Fp6Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk32fp6bf16-rocdlcvtscalef32srpk32fp6bf16op)
  - [`rocdl.cvt.scalef32.sr.pk32.fp6.f16`(ROCDL::CvtScaleF32SrPk32Fp6F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk32fp6f16-rocdlcvtscalef32srpk32fp6f16op)
  - [`rocdl.cvt.scalef32.sr.pk32.fp6.f32`(ROCDL::CvtScaleF32SrPk32Fp6F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk32fp6f32-rocdlcvtscalef32srpk32fp6f32op)
  - [`rocdl.cvt.scalef32.sr.pk8.bf8.bf16`(ROCDL::CvtScaleF32SrPk8Bf8Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk8bf8bf16-rocdlcvtscalef32srpk8bf8bf16op)
  - [`rocdl.cvt.scalef32.sr.pk8.bf8.f16`(ROCDL::CvtScaleF32SrPk8Bf8F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk8bf8f16-rocdlcvtscalef32srpk8bf8f16op)
  - [`rocdl.cvt.scalef32.sr.pk8.bf8.f32`(ROCDL::CvtScaleF32SrPk8Bf8F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk8bf8f32-rocdlcvtscalef32srpk8bf8f32op)
  - [`rocdl.cvt.scalef32.sr.pk8.fp4.bf16`(ROCDL::CvtScaleF32SrPk8Fp4Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk8fp4bf16-rocdlcvtscalef32srpk8fp4bf16op)
  - [`rocdl.cvt.scalef32.sr.pk8.fp4.f16`(ROCDL::CvtScaleF32SrPk8Fp4F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk8fp4f16-rocdlcvtscalef32srpk8fp4f16op)
  - [`rocdl.cvt.scalef32.sr.pk8.fp4.f32`(ROCDL::CvtScaleF32SrPk8Fp4F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk8fp4f32-rocdlcvtscalef32srpk8fp4f32op)
  - [`rocdl.cvt.scalef32.sr.pk8.fp8.bf16`(ROCDL::CvtScaleF32SrPk8Fp8Bf16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk8fp8bf16-rocdlcvtscalef32srpk8fp8bf16op)
  - [`rocdl.cvt.scalef32.sr.pk8.fp8.f16`(ROCDL::CvtScaleF32SrPk8Fp8F16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk8fp8f16-rocdlcvtscalef32srpk8fp8f16op)
  - [`rocdl.cvt.scalef32.sr.pk8.fp8.f32`(ROCDL::CvtScaleF32SrPk8Fp8F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtscalef32srpk8fp8f32-rocdlcvtscalef32srpk8fp8f32op)
  - [`rocdl.cvt.sr.bf8.f32`(ROCDL::CvtSrBf8F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtsrbf8f32-rocdlcvtsrbf8f32op)
  - [`rocdl.cvt.sr.fp8.f32`(ROCDL::CvtSrFp8F32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlcvtsrfp8f32-rocdlcvtsrfp8f32op)
  - [`rocdl.ds.load.tr16.b128`(ROCDL::DsLoadTr16_B128)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdldsloadtr16b128-rocdldsloadtr16_b128)
  - [`rocdl.ds.load.tr4.b64`(ROCDL::DsLoadTr4_B64)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdldsloadtr4b64-rocdldsloadtr4_b64)
  - [`rocdl.ds.load.tr6.b96`(ROCDL::DsLoadTr6_B96)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdldsloadtr6b96-rocdldsloadtr6_b96)
  - [`rocdl.ds.load.tr8.b64`(ROCDL::DsLoadTr8_B64)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdldsloadtr8b64-rocdldsloadtr8_b64)
  - [`rocdl.ds.read.tr16.b64`(ROCDL::ds_read_tr16_b64)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdldsreadtr16b64-rocdlds_read_tr16_b64)
  - [`rocdl.ds.read.tr4.b64`(ROCDL::ds_read_tr4_b64)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdldsreadtr4b64-rocdlds_read_tr4_b64)
  - [`rocdl.ds.read.tr6.b96`(ROCDL::ds_read_tr6_b96)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdldsreadtr6b96-rocdlds_read_tr6_b96)
  - [`rocdl.ds.read.tr8.b64`(ROCDL::ds_read_tr8_b64)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdldsreadtr8b64-rocdlds_read_tr8_b64)
  - [`rocdl.ds_bpermute`(ROCDL::DsBpermuteOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlds_bpermute-rocdldsbpermuteop)
  - [`rocdl.ds_swizzle`(ROCDL::DsSwizzleOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlds_swizzle-rocdldsswizzleop)
  - [`rocdl.fmed3`(ROCDL::FMed3Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlfmed3-rocdlfmed3op)
  - [`rocdl.global.load.async.to.lds.b128`(ROCDL::GlobalLoadAsyncToLDSB128Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlgloballoadasynctoldsb128-rocdlgloballoadasynctoldsb128op)
  - [`rocdl.global.load.async.to.lds.b32`(ROCDL::GlobalLoadAsyncToLDSB32Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlgloballoadasynctoldsb32-rocdlgloballoadasynctoldsb32op)
  - [`rocdl.global.load.async.to.lds.b64`(ROCDL::GlobalLoadAsyncToLDSB64Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlgloballoadasynctoldsb64-rocdlgloballoadasynctoldsb64op)
  - [`rocdl.global.load.async.to.lds.b8`(ROCDL::GlobalLoadAsyncToLDSB8Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlgloballoadasynctoldsb8-rocdlgloballoadasynctoldsb8op)
  - [`rocdl.global.load.lds`(ROCDL::GlobalLoadLDSOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlgloballoadlds-rocdlgloballoadldsop)
  - [`rocdl.global.load.tr.b128`(ROCDL::GlobalLoadTr8_B128)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlgloballoadtrb128-rocdlgloballoadtr8_b128)
  - [`rocdl.global.load.tr.b64`(ROCDL::GlobalLoadTr8_B64)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlgloballoadtrb64-rocdlgloballoadtr8_b64)
  - [`rocdl.global.load.tr4.b64`(ROCDL::GlobalLoadTr4_B64)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlgloballoadtr4b64-rocdlgloballoadtr4_b64)
  - [`rocdl.global.load.tr6.b96`(ROCDL::GlobalLoadTr6_B96)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlgloballoadtr6b96-rocdlgloballoadtr6_b96)
  - [`rocdl.grid.dim.x`(ROCDL::GridDimXOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlgriddimx-rocdlgriddimxop)
  - [`rocdl.grid.dim.y`(ROCDL::GridDimYOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlgriddimy-rocdlgriddimyop)
  - [`rocdl.grid.dim.z`(ROCDL::GridDimZOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlgriddimz-rocdlgriddimzop)
  - [`rocdl.iglp.opt`(ROCDL::IglpOpt)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdliglpopt-rocdliglpopt)
  - [`rocdl.load.to.lds`(ROCDL::LoadToLDSOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlloadtolds-rocdlloadtoldsop)
  - [`rocdl.make.buffer.rsrc`(ROCDL::MakeBufferRsrcOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmakebufferrsrc-rocdlmakebufferrsrcop)
  - [`rocdl.mbcnt.hi`(ROCDL::MbcntHiOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmbcnthi-rocdlmbcnthiop)
  - [`rocdl.mbcnt.lo`(ROCDL::MbcntLoOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmbcntlo-rocdlmbcntloop)
  - [`rocdl.mfma.f32.16x16x16bf16.1k`(ROCDL::mfma_f32_16x16x16bf16_1k)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3216x16x16bf161k-rocdlmfma_f32_16x16x16bf16_1k)
  - [`rocdl.mfma.f32.16x16x16f16`(ROCDL::mfma_f32_16x16x16f16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3216x16x16f16-rocdlmfma_f32_16x16x16f16)
  - [`rocdl.mfma.f32.16x16x1f32`(ROCDL::mfma_f32_16x16x1f32)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3216x16x1f32-rocdlmfma_f32_16x16x1f32)
  - [`rocdl.mfma.f32.16x16x2bf16`(ROCDL::mfma_f32_16x16x2bf16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3216x16x2bf16-rocdlmfma_f32_16x16x2bf16)
  - [`rocdl.mfma.f32.16x16x32.bf16`(ROCDL::mfma_f32_16x16x32_bf16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3216x16x32bf16-rocdlmfma_f32_16x16x32_bf16)
  - [`rocdl.mfma.f32.16x16x32.bf8.bf8`(ROCDL::mfma_f32_16x16x32_bf8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3216x16x32bf8bf8-rocdlmfma_f32_16x16x32_bf8_bf8)
  - [`rocdl.mfma.f32.16x16x32.bf8.fp8`(ROCDL::mfma_f32_16x16x32_bf8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3216x16x32bf8fp8-rocdlmfma_f32_16x16x32_bf8_fp8)
  - [`rocdl.mfma.f32.16x16x32.f16`(ROCDL::mfma_f32_16x16x32_f16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3216x16x32f16-rocdlmfma_f32_16x16x32_f16)
  - [`rocdl.mfma.f32.16x16x32.fp8.bf8`(ROCDL::mfma_f32_16x16x32_fp8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3216x16x32fp8bf8-rocdlmfma_f32_16x16x32_fp8_bf8)
  - [`rocdl.mfma.f32.16x16x32.fp8.fp8`(ROCDL::mfma_f32_16x16x32_fp8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3216x16x32fp8fp8-rocdlmfma_f32_16x16x32_fp8_fp8)
  - [`rocdl.mfma.f32.16x16x4bf16.1k`(ROCDL::mfma_f32_16x16x4bf16_1k)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3216x16x4bf161k-rocdlmfma_f32_16x16x4bf16_1k)
  - [`rocdl.mfma.f32.16x16x4f16`(ROCDL::mfma_f32_16x16x4f16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3216x16x4f16-rocdlmfma_f32_16x16x4f16)
  - [`rocdl.mfma.f32.16x16x4f32`(ROCDL::mfma_f32_16x16x4f32)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3216x16x4f32-rocdlmfma_f32_16x16x4f32)
  - [`rocdl.mfma.f32.16x16x8.xf32`(ROCDL::mfma_f32_16x16x8_xf32)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3216x16x8xf32-rocdlmfma_f32_16x16x8_xf32)
  - [`rocdl.mfma.f32.16x16x8bf16`(ROCDL::mfma_f32_16x16x8bf16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3216x16x8bf16-rocdlmfma_f32_16x16x8bf16)
  - [`rocdl.mfma.f32.32x32x16.bf16`(ROCDL::mfma_f32_32x32x16_bf16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3232x32x16bf16-rocdlmfma_f32_32x32x16_bf16)
  - [`rocdl.mfma.f32.32x32x16.bf8.bf8`(ROCDL::mfma_f32_32x32x16_bf8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3232x32x16bf8bf8-rocdlmfma_f32_32x32x16_bf8_bf8)
  - [`rocdl.mfma.f32.32x32x16.bf8.fp8`(ROCDL::mfma_f32_32x32x16_bf8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3232x32x16bf8fp8-rocdlmfma_f32_32x32x16_bf8_fp8)
  - [`rocdl.mfma.f32.32x32x16.f16`(ROCDL::mfma_f32_32x32x16_f16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3232x32x16f16-rocdlmfma_f32_32x32x16_f16)
  - [`rocdl.mfma.f32.32x32x16.fp8.bf8`(ROCDL::mfma_f32_32x32x16_fp8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3232x32x16fp8bf8-rocdlmfma_f32_32x32x16_fp8_bf8)
  - [`rocdl.mfma.f32.32x32x16.fp8.fp8`(ROCDL::mfma_f32_32x32x16_fp8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3232x32x16fp8fp8-rocdlmfma_f32_32x32x16_fp8_fp8)
  - [`rocdl.mfma.f32.32x32x1f32`(ROCDL::mfma_f32_32x32x1f32)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3232x32x1f32-rocdlmfma_f32_32x32x1f32)
  - [`rocdl.mfma.f32.32x32x2bf16`(ROCDL::mfma_f32_32x32x2bf16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3232x32x2bf16-rocdlmfma_f32_32x32x2bf16)
  - [`rocdl.mfma.f32.32x32x2f32`(ROCDL::mfma_f32_32x32x2f32)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3232x32x2f32-rocdlmfma_f32_32x32x2f32)
  - [`rocdl.mfma.f32.32x32x4.xf32`(ROCDL::mfma_f32_32x32x4_xf32)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3232x32x4xf32-rocdlmfma_f32_32x32x4_xf32)
  - [`rocdl.mfma.f32.32x32x4bf16`(ROCDL::mfma_f32_32x32x4bf16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3232x32x4bf16-rocdlmfma_f32_32x32x4bf16)
  - [`rocdl.mfma.f32.32x32x4bf16.1k`(ROCDL::mfma_f32_32x32x4bf16_1k)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3232x32x4bf161k-rocdlmfma_f32_32x32x4bf16_1k)
  - [`rocdl.mfma.f32.32x32x4f16`(ROCDL::mfma_f32_32x32x4f16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3232x32x4f16-rocdlmfma_f32_32x32x4f16)
  - [`rocdl.mfma.f32.32x32x8bf16.1k`(ROCDL::mfma_f32_32x32x8bf16_1k)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3232x32x8bf161k-rocdlmfma_f32_32x32x8bf16_1k)
  - [`rocdl.mfma.f32.32x32x8f16`(ROCDL::mfma_f32_32x32x8f16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf3232x32x8f16-rocdlmfma_f32_32x32x8f16)
  - [`rocdl.mfma.f32.4x4x1f32`(ROCDL::mfma_f32_4x4x1f32)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf324x4x1f32-rocdlmfma_f32_4x4x1f32)
  - [`rocdl.mfma.f32.4x4x2bf16`(ROCDL::mfma_f32_4x4x2bf16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf324x4x2bf16-rocdlmfma_f32_4x4x2bf16)
  - [`rocdl.mfma.f32.4x4x4bf16.1k`(ROCDL::mfma_f32_4x4x4bf16_1k)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf324x4x4bf161k-rocdlmfma_f32_4x4x4bf16_1k)
  - [`rocdl.mfma.f32.4x4x4f16`(ROCDL::mfma_f32_4x4x4f16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf324x4x4f16-rocdlmfma_f32_4x4x4f16)
  - [`rocdl.mfma.f64.16x16x4f64`(ROCDL::mfma_f64_16x16x4f64)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf6416x16x4f64-rocdlmfma_f64_16x16x4f64)
  - [`rocdl.mfma.f64.4x4x4f64`(ROCDL::mfma_f64_4x4x4f64)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmaf644x4x4f64-rocdlmfma_f64_4x4x4f64)
  - [`rocdl.mfma.i32.16x16x16i8`(ROCDL::mfma_i32_16x16x16i8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmai3216x16x16i8-rocdlmfma_i32_16x16x16i8)
  - [`rocdl.mfma.i32.16x16x32.i8`(ROCDL::mfma_i32_16x16x32_i8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmai3216x16x32i8-rocdlmfma_i32_16x16x32_i8)
  - [`rocdl.mfma.i32.16x16x4i8`(ROCDL::mfma_i32_16x16x4i8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmai3216x16x4i8-rocdlmfma_i32_16x16x4i8)
  - [`rocdl.mfma.i32.16x16x64.i8`(ROCDL::mfma_i32_16x16x64_i8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmai3216x16x64i8-rocdlmfma_i32_16x16x64_i8)
  - [`rocdl.mfma.i32.32x32x16.i8`(ROCDL::mfma_i32_32x32x16_i8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmai3232x32x16i8-rocdlmfma_i32_32x32x16_i8)
  - [`rocdl.mfma.i32.32x32x32.i8`(ROCDL::mfma_i32_32x32x32_i8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmai3232x32x32i8-rocdlmfma_i32_32x32x32_i8)
  - [`rocdl.mfma.i32.32x32x4i8`(ROCDL::mfma_i32_32x32x4i8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmai3232x32x4i8-rocdlmfma_i32_32x32x4i8)
  - [`rocdl.mfma.i32.32x32x8i8`(ROCDL::mfma_i32_32x32x8i8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmai3232x32x8i8-rocdlmfma_i32_32x32x8i8)
  - [`rocdl.mfma.i32.4x4x4i8`(ROCDL::mfma_i32_4x4x4i8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmai324x4x4i8-rocdlmfma_i32_4x4x4i8)
  - [`rocdl.mfma.scale.f32.16x16x128.f8f6f4`(ROCDL::mfma_scale_f32_16x16x128_f8f6f4)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmascalef3216x16x128f8f6f4-rocdlmfma_scale_f32_16x16x128_f8f6f4)
  - [`rocdl.mfma.scale.f32.32x32x64.f8f6f4`(ROCDL::mfma_scale_f32_32x32x64_f8f6f4)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlmfmascalef3232x32x64f8f6f4-rocdlmfma_scale_f32_32x32x64_f8f6f4)
  - [`rocdl.permlane16.swap`(ROCDL::Permlane16SwapOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlpermlane16swap-rocdlpermlane16swapop)
  - [`rocdl.permlane32.swap`(ROCDL::Permlane32SwapOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlpermlane32swap-rocdlpermlane32swapop)
  - [`rocdl.permlanex16`(ROCDL::PermlaneX16Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlpermlanex16-rocdlpermlanex16op)
  - [`rocdl.raw.buffer.atomic.cmpswap`(ROCDL::RawBufferAtomicCmpSwap)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlrawbufferatomiccmpswap-rocdlrawbufferatomiccmpswap)
  - [`rocdl.raw.buffer.atomic.fadd`(ROCDL::RawBufferAtomicFAddOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlrawbufferatomicfadd-rocdlrawbufferatomicfaddop)
  - [`rocdl.raw.buffer.atomic.fmax`(ROCDL::RawBufferAtomicFMaxOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlrawbufferatomicfmax-rocdlrawbufferatomicfmaxop)
  - [`rocdl.raw.buffer.atomic.smax`(ROCDL::RawBufferAtomicSMaxOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlrawbufferatomicsmax-rocdlrawbufferatomicsmaxop)
  - [`rocdl.raw.buffer.atomic.umin`(ROCDL::RawBufferAtomicUMinOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlrawbufferatomicumin-rocdlrawbufferatomicuminop)
  - [`rocdl.raw.buffer.load`(ROCDL::RawBufferLoadOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlrawbufferload-rocdlrawbufferloadop)
  - [`rocdl.raw.buffer.store`(ROCDL::RawBufferStoreOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlrawbufferstore-rocdlrawbufferstoreop)
  - [`rocdl.raw.ptr.buffer.atomic.cmpswap`(ROCDL::RawPtrBufferAtomicCmpSwap)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlrawptrbufferatomiccmpswap-rocdlrawptrbufferatomiccmpswap)
  - [`rocdl.raw.ptr.buffer.atomic.fadd`(ROCDL::RawPtrBufferAtomicFaddOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlrawptrbufferatomicfadd-rocdlrawptrbufferatomicfaddop)
  - [`rocdl.raw.ptr.buffer.atomic.fmax`(ROCDL::RawPtrBufferAtomicFmaxOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlrawptrbufferatomicfmax-rocdlrawptrbufferatomicfmaxop)
  - [`rocdl.raw.ptr.buffer.atomic.smax`(ROCDL::RawPtrBufferAtomicSmaxOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlrawptrbufferatomicsmax-rocdlrawptrbufferatomicsmaxop)
  - [`rocdl.raw.ptr.buffer.atomic.umin`(ROCDL::RawPtrBufferAtomicUminOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlrawptrbufferatomicumin-rocdlrawptrbufferatomicuminop)
  - [`rocdl.raw.ptr.buffer.load`(ROCDL::RawPtrBufferLoadOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlrawptrbufferload-rocdlrawptrbufferloadop)
  - [`rocdl.raw.ptr.buffer.load.lds`(ROCDL::RawPtrBufferLoadLdsOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlrawptrbufferloadlds-rocdlrawptrbufferloadldsop)
  - [`rocdl.raw.ptr.buffer.store`(ROCDL::RawPtrBufferStoreOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlrawptrbufferstore-rocdlrawptrbufferstoreop)
  - [`rocdl.readfirstlane`(ROCDL::ReadfirstlaneOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlreadfirstlane-rocdlreadfirstlaneop)
  - [`rocdl.readlane`(ROCDL::ReadlaneOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlreadlane-rocdlreadlaneop)
  - [`rocdl.s.barrier`(ROCDL::SBarrierOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsbarrier-rocdlsbarrierop)
  - [`rocdl.s.barrier.init`(ROCDL::BarrierInitOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsbarrierinit-rocdlbarrierinitop)
  - [`rocdl.s.barrier.join`(ROCDL::BarrierJoinOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsbarrierjoin-rocdlbarrierjoinop)
  - [`rocdl.s.barrier.leave`(ROCDL::BarrierLeaveOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsbarrierleave-rocdlbarrierleaveop)
  - [`rocdl.s.barrier.signal`(ROCDL::BarrierSignalOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsbarriersignal-rocdlbarriersignalop)
  - [`rocdl.s.barrier.signal.isfirst`(ROCDL::BarrierSignalIsfirstOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsbarriersignalisfirst-rocdlbarriersignalisfirstop)
  - [`rocdl.s.barrier.signal.var`(ROCDL::BarrierSignalVarOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsbarriersignalvar-rocdlbarriersignalvarop)
  - [`rocdl.s.barrier.wait`(ROCDL::BarrierWaitOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsbarrierwait-rocdlbarrierwaitop)
  - [`rocdl.s.get.barrier.state`(ROCDL::GetBarrierStateOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsgetbarrierstate-rocdlgetbarrierstateop)
  - [`rocdl.s.get.named.barrier.state`(ROCDL::GetNamedBarrierStateOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsgetnamedbarrierstate-rocdlgetnamedbarrierstateop)
  - [`rocdl.s.setprio`(ROCDL::SetPrioOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlssetprio-rocdlsetprioop)
  - [`rocdl.s.sleep`(ROCDL::SSleepOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlssleep-rocdlssleepop)
  - [`rocdl.s.wait.asynccnt`(ROCDL::WaitAsynccntOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlswaitasynccnt-rocdlwaitasynccntop)
  - [`rocdl.s.wait.dscnt`(ROCDL::WaitDscntOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlswaitdscnt-rocdlwaitdscntop)
  - [`rocdl.s.wait.expcnt`(ROCDL::WaitExpcntOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlswaitexpcnt-rocdlwaitexpcntop)
  - [`rocdl.s.wait.loadcnt`(ROCDL::WaitLoadcntOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlswaitloadcnt-rocdlwaitloadcntop)
  - [`rocdl.s.wait.storecnt`(ROCDL::WaitStorecntOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlswaitstorecnt-rocdlwaitstorecntop)
  - [`rocdl.s.wait.tensorcnt`(ROCDL::WaitTensorcntOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlswaittensorcnt-rocdlwaittensorcntop)
  - [`rocdl.s.waitcnt`(ROCDL::SWaitcntOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlswaitcnt-rocdlswaitcntop)
  - [`rocdl.sched.barrier`(ROCDL::SchedBarrier)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlschedbarrier-rocdlschedbarrier)
  - [`rocdl.sched.group.barrier`(ROCDL::SchedGroupBarrier)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlschedgroupbarrier-rocdlschedgroupbarrier)
  - [`rocdl.smfmac.f32.16x16x32.bf16`(ROCDL::smfmac_f32_16x16x32_bf16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsmfmacf3216x16x32bf16-rocdlsmfmac_f32_16x16x32_bf16)
  - [`rocdl.smfmac.f32.16x16x32.f16`(ROCDL::smfmac_f32_16x16x32_f16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsmfmacf3216x16x32f16-rocdlsmfmac_f32_16x16x32_f16)
  - [`rocdl.smfmac.f32.16x16x64.bf8.bf8`(ROCDL::smfmac_f32_16x16x64_bf8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsmfmacf3216x16x64bf8bf8-rocdlsmfmac_f32_16x16x64_bf8_bf8)
  - [`rocdl.smfmac.f32.16x16x64.bf8.fp8`(ROCDL::smfmac_f32_16x16x64_bf8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsmfmacf3216x16x64bf8fp8-rocdlsmfmac_f32_16x16x64_bf8_fp8)
  - [`rocdl.smfmac.f32.16x16x64.fp8.bf8`(ROCDL::smfmac_f32_16x16x64_fp8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsmfmacf3216x16x64fp8bf8-rocdlsmfmac_f32_16x16x64_fp8_bf8)
  - [`rocdl.smfmac.f32.16x16x64.fp8.fp8`(ROCDL::smfmac_f32_16x16x64_fp8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsmfmacf3216x16x64fp8fp8-rocdlsmfmac_f32_16x16x64_fp8_fp8)
  - [`rocdl.smfmac.f32.32x32x16.bf16`(ROCDL::smfmac_f32_32x32x16_bf16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsmfmacf3232x32x16bf16-rocdlsmfmac_f32_32x32x16_bf16)
  - [`rocdl.smfmac.f32.32x32x16.f16`(ROCDL::smfmac_f32_32x32x16_f16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsmfmacf3232x32x16f16-rocdlsmfmac_f32_32x32x16_f16)
  - [`rocdl.smfmac.f32.32x32x32.bf8.bf8`(ROCDL::smfmac_f32_32x32x32_bf8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsmfmacf3232x32x32bf8bf8-rocdlsmfmac_f32_32x32x32_bf8_bf8)
  - [`rocdl.smfmac.f32.32x32x32.bf8.fp8`(ROCDL::smfmac_f32_32x32x32_bf8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsmfmacf3232x32x32bf8fp8-rocdlsmfmac_f32_32x32x32_bf8_fp8)
  - [`rocdl.smfmac.f32.32x32x32.fp8.bf8`(ROCDL::smfmac_f32_32x32x32_fp8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsmfmacf3232x32x32fp8bf8-rocdlsmfmac_f32_32x32x32_fp8_bf8)
  - [`rocdl.smfmac.f32.32x32x32.fp8.fp8`(ROCDL::smfmac_f32_32x32x32_fp8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsmfmacf3232x32x32fp8fp8-rocdlsmfmac_f32_32x32x32_fp8_fp8)
  - [`rocdl.smfmac.i32.16x16x64.i8`(ROCDL::smfmac_i32_16x16x64_i8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsmfmaci3216x16x64i8-rocdlsmfmac_i32_16x16x64_i8)
  - [`rocdl.smfmac.i32.32x32x32.i8`(ROCDL::smfmac_i32_32x32x32_i8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlsmfmaci3232x32x32i8-rocdlsmfmac_i32_32x32x32_i8)
  - [`rocdl.tensor.load.to.lds`(ROCDL::TensorLoadToLDSOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdltensorloadtolds-rocdltensorloadtoldsop)
  - [`rocdl.tensor.load.to.lds.d2`(ROCDL::TensorLoadToLDSD2Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdltensorloadtoldsd2-rocdltensorloadtoldsd2op)
  - [`rocdl.tensor.store.from.lds`(ROCDL::TensorStoreFromLDSOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdltensorstorefromlds-rocdltensorstorefromldsop)
  - [`rocdl.tensor.store.from.lds.d2`(ROCDL::TensorStoreFromLDSD2Op)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdltensorstorefromldsd2-rocdltensorstorefromldsd2op)
  - [`rocdl.update.dpp`(ROCDL::DPPUpdateOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlupdatedpp-rocdldppupdateop)
  - [`rocdl.wavefrontsize`(ROCDL::WavefrontSizeOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwavefrontsize-rocdlwavefrontsizeop)
  - [`rocdl.wmma.bf16.16x16x16.bf16`(ROCDL::wmma_bf16_16x16x16_bf16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmabf1616x16x16bf16-rocdlwmma_bf16_16x16x16_bf16)
  - [`rocdl.wmma.bf16.16x16x32.bf16`(ROCDL::wmma_bf16_16x16x32_bf16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmabf1616x16x32bf16-rocdlwmma_bf16_16x16x32_bf16)
  - [`rocdl.wmma.bf16f32.16x16x32.bf16`(ROCDL::wmma_bf16f32_16x16x32_bf16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmabf16f3216x16x32bf16-rocdlwmma_bf16f32_16x16x32_bf16)
  - [`rocdl.wmma.f16.16x16x128.bf8_bf8`(ROCDL::wmma_f16_16x16x128_bf8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf1616x16x128bf8_bf8-rocdlwmma_f16_16x16x128_bf8_bf8)
  - [`rocdl.wmma.f16.16x16x128.bf8_fp8`(ROCDL::wmma_f16_16x16x128_bf8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf1616x16x128bf8_fp8-rocdlwmma_f16_16x16x128_bf8_fp8)
  - [`rocdl.wmma.f16.16x16x128.fp8_bf8`(ROCDL::wmma_f16_16x16x128_fp8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf1616x16x128fp8_bf8-rocdlwmma_f16_16x16x128_fp8_bf8)
  - [`rocdl.wmma.f16.16x16x128.fp8_fp8`(ROCDL::wmma_f16_16x16x128_fp8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf1616x16x128fp8_fp8-rocdlwmma_f16_16x16x128_fp8_fp8)
  - [`rocdl.wmma.f16.16x16x16.f16`(ROCDL::wmma_f16_16x16x16_f16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf1616x16x16f16-rocdlwmma_f16_16x16x16_f16)
  - [`rocdl.wmma.f16.16x16x32.f16`(ROCDL::wmma_f16_16x16x32_f16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf1616x16x32f16-rocdlwmma_f16_16x16x32_f16)
  - [`rocdl.wmma.f16.16x16x64.bf8_bf8`(ROCDL::wmma_f16_16x16x64_bf8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf1616x16x64bf8_bf8-rocdlwmma_f16_16x16x64_bf8_bf8)
  - [`rocdl.wmma.f16.16x16x64.bf8_fp8`(ROCDL::wmma_f16_16x16x64_bf8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf1616x16x64bf8_fp8-rocdlwmma_f16_16x16x64_bf8_fp8)
  - [`rocdl.wmma.f16.16x16x64.fp8_bf8`(ROCDL::wmma_f16_16x16x64_fp8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf1616x16x64fp8_bf8-rocdlwmma_f16_16x16x64_fp8_bf8)
  - [`rocdl.wmma.f16.16x16x64.fp8_fp8`(ROCDL::wmma_f16_16x16x64_fp8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf1616x16x64fp8_fp8-rocdlwmma_f16_16x16x64_fp8_fp8)
  - [`rocdl.wmma.f32.16x16x128.bf8_bf8`(ROCDL::wmma_f32_16x16x128_bf8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x128bf8_bf8-rocdlwmma_f32_16x16x128_bf8_bf8)
  - [`rocdl.wmma.f32.16x16x128.bf8_fp8`(ROCDL::wmma_f32_16x16x128_bf8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x128bf8_fp8-rocdlwmma_f32_16x16x128_bf8_fp8)
  - [`rocdl.wmma.f32.16x16x128.fp8_bf8`(ROCDL::wmma_f32_16x16x128_fp8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x128fp8_bf8-rocdlwmma_f32_16x16x128_fp8_bf8)
  - [`rocdl.wmma.f32.16x16x128.fp8_fp8`(ROCDL::wmma_f32_16x16x128_fp8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x128fp8_fp8-rocdlwmma_f32_16x16x128_fp8_fp8)
  - [`rocdl.wmma.f32.16x16x16.bf16`(ROCDL::wmma_f32_16x16x16_bf16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x16bf16-rocdlwmma_f32_16x16x16_bf16)
  - [`rocdl.wmma.f32.16x16x16.bf8_bf8`(ROCDL::wmma_f32_16x16x16_bf8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x16bf8_bf8-rocdlwmma_f32_16x16x16_bf8_bf8)
  - [`rocdl.wmma.f32.16x16x16.bf8_fp8`(ROCDL::wmma_f32_16x16x16_bf8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x16bf8_fp8-rocdlwmma_f32_16x16x16_bf8_fp8)
  - [`rocdl.wmma.f32.16x16x16.f16`(ROCDL::wmma_f32_16x16x16_f16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x16f16-rocdlwmma_f32_16x16x16_f16)
  - [`rocdl.wmma.f32.16x16x16.fp8_bf8`(ROCDL::wmma_f32_16x16x16_fp8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x16fp8_bf8-rocdlwmma_f32_16x16x16_fp8_bf8)
  - [`rocdl.wmma.f32.16x16x16.fp8_fp8`(ROCDL::wmma_f32_16x16x16_fp8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x16fp8_fp8-rocdlwmma_f32_16x16x16_fp8_fp8)
  - [`rocdl.wmma.f32.16x16x32.bf16`(ROCDL::wmma_f32_16x16x32_bf16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x32bf16-rocdlwmma_f32_16x16x32_bf16)
  - [`rocdl.wmma.f32.16x16x32.f16`(ROCDL::wmma_f32_16x16x32_f16)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x32f16-rocdlwmma_f32_16x16x32_f16)
  - [`rocdl.wmma.f32.16x16x4.f32`(ROCDL::wmma_f32_16x16x4_f32)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x4f32-rocdlwmma_f32_16x16x4_f32)
  - [`rocdl.wmma.f32.16x16x64.bf8_bf8`(ROCDL::wmma_f32_16x16x64_bf8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x64bf8_bf8-rocdlwmma_f32_16x16x64_bf8_bf8)
  - [`rocdl.wmma.f32.16x16x64.bf8_fp8`(ROCDL::wmma_f32_16x16x64_bf8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x64bf8_fp8-rocdlwmma_f32_16x16x64_bf8_fp8)
  - [`rocdl.wmma.f32.16x16x64.fp8_bf8`(ROCDL::wmma_f32_16x16x64_fp8_bf8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x64fp8_bf8-rocdlwmma_f32_16x16x64_fp8_bf8)
  - [`rocdl.wmma.f32.16x16x64.fp8_fp8`(ROCDL::wmma_f32_16x16x64_fp8_fp8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmaf3216x16x64fp8_fp8-rocdlwmma_f32_16x16x64_fp8_fp8)
  - [`rocdl.wmma.i32.16x16x16.iu4`(ROCDL::wmma_i32_16x16x16_iu4)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmai3216x16x16iu4-rocdlwmma_i32_16x16x16_iu4)
  - [`rocdl.wmma.i32.16x16x16.iu8`(ROCDL::wmma_i32_16x16x16_iu8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmai3216x16x16iu8-rocdlwmma_i32_16x16x16_iu8)
  - [`rocdl.wmma.i32.16x16x32.iu4`(ROCDL::wmma_i32_16x16x32_iu4)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmai3216x16x32iu4-rocdlwmma_i32_16x16x32_iu4)
  - [`rocdl.wmma.i32.16x16x64.iu8`(ROCDL::wmma_i32_16x16x64_iu8)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmai3216x16x64iu8-rocdlwmma_i32_16x16x64_iu8)
  - [`rocdl.wmma.scale.f32.16x16x128.f8f6f4`(ROCDL::wmma_scale_f32_16x16x128_f8f6f4)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmascalef3216x16x128f8f6f4-rocdlwmma_scale_f32_16x16x128_f8f6f4)
  - [`rocdl.wmma.scale.f32.32x16x128.f4`(ROCDL::wmma_scale_f32_32x16x128_f4)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmascalef3232x16x128f4-rocdlwmma_scale_f32_32x16x128_f4)
  - [`rocdl.wmma.scale16.f32.16x16x128.f8f6f4`(ROCDL::wmma_scale16_f32_16x16x128_f8f6f4)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmascale16f3216x16x128f8f6f4-rocdlwmma_scale16_f32_16x16x128_f8f6f4)
  - [`rocdl.wmma.scale16.f32.32x16x128.f4`(ROCDL::wmma_scale16_f32_32x16x128_f4)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlwmmascale16f3232x16x128f4-rocdlwmma_scale16_f32_32x16x128_f4)
  - [`rocdl.workgroup.dim.x`(ROCDL::BlockDimXOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlworkgroupdimx-rocdlblockdimxop)
  - [`rocdl.workgroup.dim.y`(ROCDL::BlockDimYOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlworkgroupdimy-rocdlblockdimyop)
  - [`rocdl.workgroup.dim.z`(ROCDL::BlockDimZOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlworkgroupdimz-rocdlblockdimzop)
  - [`rocdl.workgroup.id.x`(ROCDL::BlockIdXOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlworkgroupidx-rocdlblockidxop)
  - [`rocdl.workgroup.id.y`(ROCDL::BlockIdYOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlworkgroupidy-rocdlblockidyop)
  - [`rocdl.workgroup.id.z`(ROCDL::BlockIdZOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlworkgroupidz-rocdlblockidzop)
  - [`rocdl.workitem.id.x`(ROCDL::ThreadIdXOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlworkitemidx-rocdlthreadidxop)
  - [`rocdl.workitem.id.y`(ROCDL::ThreadIdYOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlworkitemidy-rocdlthreadidyop)
  - [`rocdl.workitem.id.z`(ROCDL::ThreadIdZOp)](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdlworkitemidz-rocdlthreadidzop)
- [属性](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#attributes-161)
  - [ROCDLTargetAttr](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/#rocdltargetattr)

## 操作

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/LLVMIR/ROCDLOps.td)

### `rocdl.ballot`(ROCDL::BallotOp)

*跨线程组表决*

语法：

```
operation ::= `rocdl.ballot` $pred attr-dict `:` type($res)
```

Ballot 提供一个位掩码，其中包含来自每个lane的 1 位谓词值。结果的第 n 位包含由第 n 个线程束lane贡献的 1 位。

#### 操作数：

| Operand | Description            |
| :-----: | ---------------------- |
| `pred`  | 1-bit signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.barrier`(ROCDL::BarrierOp)

语法：

```
operation ::= `rocdl.barrier` attr-dict
```

### `rocdl.cluster.id.x`(ROCDL::ClusterIdXOp)

语法：

```
operation ::= `rocdl.cluster.id.x` (`range` $range^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                                                  |
| --------- | ------------------------------- | ------------------------------------------------------------ |
| `range`   | ::mlir::LLVM::ConstantRangeAttr | A range of two integers, corresponding to LLVM's ConstantRange`A pair of two integers, mapping to the ConstantRange structure in LLVM IR, which is allowed to wrap or be empty. The range represented is [Lower, Upper), and is either signed or unsigned depending on context. lower and upper must have the same width. Syntax: `<` `i`(width($lower)) $lower `,` $upper `>` ` |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.cluster.id.y`(ROCDL::ClusterIdYOp)

语法：

```
operation ::= `rocdl.cluster.id.y` (`range` $range^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                                                  |
| --------- | ------------------------------- | ------------------------------------------------------------ |
| `range`   | ::mlir::LLVM::ConstantRangeAttr | A range of two integers, corresponding to LLVM's ConstantRange`A pair of two integers, mapping to the ConstantRange structure in LLVM IR, which is allowed to wrap or be empty. The range represented is [Lower, Upper), and is either signed or unsigned depending on context. lower and upper must have the same width. Syntax: `<` `i`(width($lower)) $lower `,` $upper `>` ` |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.cluster.id.z`(ROCDL::ClusterIdZOp)

语法：

```
operation ::= `rocdl.cluster.id.z` (`range` $range^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                                                  |
| --------- | ------------------------------- | ------------------------------------------------------------ |
| `range`   | ::mlir::LLVM::ConstantRangeAttr | A range of two integers, corresponding to LLVM's ConstantRange`A pair of two integers, mapping to the ConstantRange structure in LLVM IR, which is allowed to wrap or be empty. The range represented is [Lower, Upper), and is either signed or unsigned depending on context. lower and upper must have the same width. Syntax: `<` `i`(width($lower)) $lower `,` $upper `>` ` |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.cluster.load.async.to.lds.b128`(ROCDL::ClusterLoadAsyncToLDSB128Op)

语法：

```
operation ::= `rocdl.cluster.load.async.to.lds.b128` $globalPtr `,`  $ldsPtr `,` $offset `,` $cpol `,` $mask
              attr-dict `:` qualified(type($globalPtr)) `,` qualified(type($ldsPtr))
```

向工作组集群广播 128 位数据的内存加载。

支持于gfx1250及以上架构。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                          |
| ---------------- | ------------------- | ------------------------------------ |
| `offset`         | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `cpol`           | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `mask`           | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array |

#### 操作数：

|   Operand   | Description                     |
| :---------: | ------------------------------- |
| `globalPtr` | LLVM pointer in address space 1 |
|  `ldsPtr`   | LLVM pointer in address space 3 |

### `rocdl.cluster.load.async.to.lds.b32`(ROCDL::ClusterLoadAsyncToLDSB32Op)

语法：

```
operation ::= `rocdl.cluster.load.async.to.lds.b32` $globalPtr `,`  $ldsPtr `,` $offset `,` $cpol `,` $mask
              attr-dict `:` qualified(type($globalPtr)) `,` qualified(type($ldsPtr))
```

向工作组集群广播 32 位数据的内存加载。

支持于gfx1250及以上版本。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                          |
| ---------------- | ------------------- | ------------------------------------ |
| `offset`         | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `cpol`           | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `mask`           | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array |

#### 操作数：

|   Operand   | Description                     |
| :---------: | ------------------------------- |
| `globalPtr` | LLVM pointer in address space 1 |
|  `ldsPtr`   | LLVM pointer in address space 3 |

### `rocdl.cluster.load.async.to.lds.b64`(ROCDL::ClusterLoadAsyncToLDSB64Op)

语法：

```
operation ::= `rocdl.cluster.load.async.to.lds.b64` $globalPtr `,`  $ldsPtr `,` $offset `,` $cpol `,` $mask
              attr-dict `:` qualified(type($globalPtr)) `,` qualified(type($ldsPtr))
```

向工作组集群广播 64 位数据的内存加载。

支持于gfx1250及以上版本。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                          |
| ---------------- | ------------------- | ------------------------------------ |
| `offset`         | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `cpol`           | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `mask`           | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array |

#### 操作数：

|   Operand   | Description                     |
| :---------: | ------------------------------- |
| `globalPtr` | LLVM pointer in address space 1 |
|  `ldsPtr`   | LLVM pointer in address space 3 |

### `rocdl.cluster.load.async.to.lds.b8`(ROCDL::ClusterLoadAsyncToLDSB8Op)

语法：

```
operation ::= `rocdl.cluster.load.async.to.lds.b8` $globalPtr `,`  $ldsPtr `,` $offset `,` $cpol `,` $mask
              attr-dict `:` qualified(type($globalPtr)) `,` qualified(type($ldsPtr))
```

向工作组集群广播 8 位数据的内存加载。

支持于gfx1250及以上版本。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                          |
| ---------------- | ------------------- | ------------------------------------ |
| `offset`         | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `cpol`           | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `mask`           | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array |

#### 操作数：

|   Operand   | Description                     |
| :---------: | ------------------------------- |
| `globalPtr` | LLVM pointer in address space 1 |
|  `ldsPtr`   | LLVM pointer in address space 3 |

### `rocdl.cvt.f32.bf8`(ROCDL::CvtF32Bf8Op)

*将bf8转换为f32*

语法：

```
operation ::= `rocdl.cvt.f32.bf8` attr-dict $srcA `[` $byteSel `]` `:` type($res)
```

将`srcA`的第`byteSel`位中的8位bf8值转换为fp32。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `byteSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
| `srcA`  | 32-bit signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.cvt.f32.fp8`(ROCDL::CvtF32Fp8Op)

*将 fp8 转换为 f32*

语法：

```
operation ::= `rocdl.cvt.f32.fp8` attr-dict $srcA `[` $byteSel `]` `:` type($res)
```

将`srcA`的第`byteSel`位处的 8 位 fp8 值转换为 fp32。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `byteSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
| `srcA`  | 32-bit signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.cvt.pk.bf8.f32`(ROCDL::CvtPkBf8F32Op)

*将两个f32转换为bf8*

语法：

```
operation ::= `rocdl.cvt.pk.bf8.f32` attr-dict $srcA `,` $srcB `->` $old `[` $wordSel `]` `:` type($res)
```

将`srcA`和`srcB`转换为 bf8 并存储到`old`的低/高字中，同时保留另一个字。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type           | Description                      |
| --------- | ------------------- | -------------------------------- |
| `wordSel` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
| `srcA`  | 32-bit float            |
| `srcB`  | 32-bit float            |
|  `old`  | 32-bit signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.cvt.pk.f32.bf8`(ROCDL::CvtPkF32Bf8Op)

*将打包的bf8转换为打包的f32*

语法：

```
operation ::= `rocdl.cvt.pk.f32.bf8` attr-dict $src `[` $wordSel `]` `:` type($res)
```

根据 $wordSel 将`src`转换为打包的 fp32,

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type           | Description                      |
| --------- | ------------------- | -------------------------------- |
| `wordSel` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `src`  | 32-bit signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.cvt.pk.f32.fp8`(ROCDL::CvtPkF32Fp8Op)

*将打包的fp8转换为打包的f32*

语法：

```
operation ::= `rocdl.cvt.pk.f32.fp8` attr-dict $src `[` $wordSel `]` `:` type($res)
```

根据 $wordSel 将`src`转换为打包的 fp32。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type           | Description                      |
| --------- | ------------------- | -------------------------------- |
| `wordSel` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `src`  | 32-bit signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.cvt.pk.fp8.f32`(ROCDL::CvtPkFp8F32Op)

*将两个f32转换为fp8*

语法：

```
operation ::= `rocdl.cvt.pk.fp8.f32` attr-dict $srcA `,` $srcB `->` $old `[` $wordSel `]` `:` type($res)
```

将`srcA`和`srcB`转换为 fp8 并存储到`old`的低/高字中，同时保留另一个字。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type           | Description                      |
| --------- | ------------------- | -------------------------------- |
| `wordSel` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
| `srcA`  | 32-bit float            |
| `srcB`  | 32-bit float            |
|  `old`  | 32-bit signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.cvt.pkrtz`(ROCDL::CvtPkRtz)

*将两个 f32 输入转换为 vector<2xf16>*

语法：

```
operation ::= `rocdl.cvt.pkrtz` attr-dict $srcA `,` $srcB `:` type($res)
```

将两个 f32 值转换为打包向量<2xf16>。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description  |
| :-----: | ------------ |
| `srcA`  | 32-bit float |
| `srcB`  | 32-bit float |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.cvt.scale.pk16.bf16.bf6`(ROCDL::CvtPkScalePk16Bf16Bf6Op)

*将16个bf6缩放并转换为16个bf16。*

语法：

```
operation ::= `rocdl.cvt.scale.pk16.bf16.bf6` attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
```

支持于 gfx1250及以上版本。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `scaleSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 3 |
| `scale` | 32-bit signless integer                                      |

#### 结果：

| Result | Description                                              |
| :----: | -------------------------------------------------------- |
| `res`  | fixed-length vector of bfloat16 type values of length 16 |

### `rocdl.cvt.scale.pk16.bf16.fp6`(ROCDL::CvtPkScalePk16Bf16Fp6Op)

*将16个fp6缩放并转换为16个bf16。*

语法：

```
operation ::= `rocdl.cvt.scale.pk16.bf16.fp6` attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
```

支持gfx1250及以上版本。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `scaleSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 3 |
| `scale` | 32-bit signless integer                                      |

#### 结果：

| Result | Description                                              |
| :----: | -------------------------------------------------------- |
| `res`  | fixed-length vector of bfloat16 type values of length 16 |

### `rocdl.cvt.scale.pk16.f16.bf6`(ROCDL::CvtPkScalePk16F16Bf6Op)

*将16个bf6缩放并转换为16个f16。*

语法：

```
operation ::= `rocdl.cvt.scale.pk16.f16.bf6` attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
```

支持于 gfx1250 及以上版本。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `scaleSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 3 |
| `scale` | 32-bit signless integer                                      |

#### 结果：

| Result | Description                                             |
| :----: | ------------------------------------------------------- |
| `res`  | fixed-length vector of 16-bit float values of length 16 |

### `rocdl.cvt.scale.pk16.f16.fp6`(ROCDL::CvtPkScalePk16F16Fp6Op)

*将 16个 fp6 缩放并转换为 16个 f16。*

语法：

```
operation ::= `rocdl.cvt.scale.pk16.f16.fp6` attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
```

支持gfx1250及以上版本。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `scaleSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 3 |
| `scale` | 32-bit signless integer                                      |

#### 结果：

| Result | Description                                             |
| :----: | ------------------------------------------------------- |
| `res`  | fixed-length vector of 16-bit float values of length 16 |

### `rocdl.cvt.scale.pk16.f32.bf6`(ROCDL::CvtPkScalePk16F32Bf6Op)

*将 16 个 bf6 缩放并转换为 16 个 f32。*

语法：

```
operation ::= `rocdl.cvt.scale.pk16.f32.bf6` attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
```

支持于 gfx1250 及以上版本。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `scaleSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 3 |
| `scale` | 32-bit signless integer                                      |

#### 结果：

| Result | Description                                             |
| :----: | ------------------------------------------------------- |
| `res`  | fixed-length vector of 32-bit float values of length 16 |

### `rocdl.cvt.scale.pk16.f32.fp6`(ROCDL::CvtPkScalePk16F32Fp6Op)

*将 16个 fp6 缩放并转换为 16个 f32。*

语法：

```
operation ::= `rocdl.cvt.scale.pk16.f32.fp6` attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
```

支持gfx1250及以上版本。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `scaleSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 3 |
| `scale` | 32-bit signless integer                                      |

#### 结果：

| Result | Description                                             |
| :----: | ------------------------------------------------------- |
| `res`  | fixed-length vector of 32-bit float values of length 16 |

### `rocdl.cvt.scale.pk8.bf16.bf8`(ROCDL::CvtPkScalePk8Bf16Bf8Op)

*将8个bf8缩放并转换为8个bf16。*

语法：

```
operation ::= `rocdl.cvt.scale.pk8.bf16.bf8` attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
```

支持于 gfx1250 及以上版本。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `scaleSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 2 |
| `scale` | 32-bit signless integer                                      |

#### 结果：

| Result | Description                                             |
| :----: | ------------------------------------------------------- |
| `res`  | fixed-length vector of bfloat16 type values of length 8 |

### `rocdl.cvt.scale.pk8.bf16.fp4`(ROCDL::CvtPkScalePk8Bf16Fp4Op)

*将 8 个 fp4 缩放并转换为 8 个 bf16。*

语法：

```
operation ::= `rocdl.cvt.scale.pk8.bf16.fp4` attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
```

支持于 gfx1250 及以上版本。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `scaleSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `src`  | 32-bit signless integer |
| `scale` | 32-bit signless integer |

#### 结果：

| Result | Description                                             |
| :----: | ------------------------------------------------------- |
| `res`  | fixed-length vector of bfloat16 type values of length 8 |

### `rocdl.cvt.scale.pk8.bf16.fp8`(ROCDL::CvtPkScalePk8Bf16Fp8Op)

*将 8个 fp8 缩放并转换为 8个 bf16。*

语法：

```
operation ::= `rocdl.cvt.scale.pk8.bf16.fp8` attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
```

支持于 gfx1250 及以上版本。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `scaleSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 2 |
| `scale` | 32-bit signless integer                                      |

#### 结果：

| Result | Description                                             |
| :----: | ------------------------------------------------------- |
| `res`  | fixed-length vector of bfloat16 type values of length 8 |

### `rocdl.cvt.scale.pk8.f16.bf8`(ROCDL::CvtPkScalePk8F16Bf8Op)

*将8个bf8缩放并转换为8个f16。*

语法：

```
operation ::= `rocdl.cvt.scale.pk8.f16.bf8` attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
```

支持于 gfx1250 及以上版本。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `scaleSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 2 |
| `scale` | 32-bit signless integer                                      |

#### 结果：

| Result | Description                                            |
| :----: | ------------------------------------------------------ |
| `res`  | fixed-length vector of 16-bit float values of length 8 |

### `rocdl.cvt.scale.pk8.f16.fp4`(ROCDL::CvtPkScalePk8F16Fp4Op)

*将 8个 fp4 缩放并转换为 8个 f16。*

语法：

```
operation ::= `rocdl.cvt.scale.pk8.f16.fp4` attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
```

支持于 gfx1250 及以上版本。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `scaleSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `src`  | 32-bit signless integer |
| `scale` | 32-bit signless integer |

#### 结果：

| Result | Description                                            |
| :----: | ------------------------------------------------------ |
| `res`  | fixed-length vector of 16-bit float values of length 8 |

### `rocdl.cvt.scale.pk8.f16.fp8`(ROCDL::CvtPkScalePk8F16Fp8Op)

*将 8个 fp8 缩放并转换为 8个 f16。*

语法：

```
operation ::= `rocdl.cvt.scale.pk8.f16.fp8` attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
```

支持于 gfx1250 及以上版本。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `scaleSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 2 |
| `scale` | 32-bit signless integer                                      |

#### 结果：

| Result | Description                                            |
| :----: | ------------------------------------------------------ |
| `res`  | fixed-length vector of 16-bit float values of length 8 |

### `rocdl.cvt.scale.pk8.f32.bf8`(ROCDL::CvtPkScalePk8F32Bf8Op)

*将8个bf8缩放并转换为8个f32。*

语法：

```
operation ::= `rocdl.cvt.scale.pk8.f32.bf8` attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
```

支持于 gfx1250 及以上版本。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `scaleSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 2 |
| `scale` | 32-bit signless integer                                      |

#### 结果：

| Result | Description                                            |
| :----: | ------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit float values of length 8 |

### `rocdl.cvt.scale.pk8.f32.fp4`(ROCDL::CvtPkScalePk8F32Fp4Op)

*将8个fp4缩放并转换为8个f32。*

语法：

```
operation ::= `rocdl.cvt.scale.pk8.f32.fp4` attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
```

支持于 gfx1250 及以上版本。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `scaleSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `src`  | 32-bit signless integer |
| `scale` | 32-bit signless integer |

#### 结果：

| Result | Description                                            |
| :----: | ------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit float values of length 8 |

### `rocdl.cvt.scale.pk8.f32.fp8`(ROCDL::CvtPkScalePk8F32Fp8Op)

*将 8个 fp8 缩放并转换为 8个 f32。*

语法：

```
operation ::= `rocdl.cvt.scale.pk8.f32.fp8` attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
```

支持于 gfx1250 及以上版本。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `scaleSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 2 |
| `scale` | 32-bit signless integer                                      |

#### 结果：

| Result | Description                                            |
| :----: | ------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit float values of length 8 |

### `rocdl.cvt.scalef32.2xpk16.bf6.f32`(ROCDL::CvtScaleF322xPk16Bf6F32Op)

*将两个 vector<16xf32> 缩放转换为32个打包的bf6*

语法：

```
operation ::= `rocdl.cvt.scalef32.2xpk16.bf6.f32` attr-dict $src0 `,` $src1 `,` $scale `:` type($res)
```

将32位单精度浮点数转换为打包的bf6，这些数值打包成两个长度为 16 的向量，这两个向量将在逻辑上连接起来。转换前需先除以`scale`的指数部分。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
| `src0`  | fixed-length vector of 32-bit float values of length 16 |
| `src1`  | fixed-length vector of 32-bit float values of length 16 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 6 |

### `rocdl.cvt.scalef32.2xpk16.fp6.f32`(ROCDL::CvtScaleF322xPk16Fp6F32Op)

*将两个 vector<16xf32> 缩放转换为32个打包的f6*

语法：

```
operation ::= `rocdl.cvt.scalef32.2xpk16.fp6.f32` attr-dict $src0 `,` $src1 `,` $scale `:` type($res)
```

将32位单精度浮点数转换为打包的f6，这些数值打包成两个长度为 16 的向量，这两个向量将在逻辑上连接起来。转换前需先除以`scale`的指数部分。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
| `src0`  | fixed-length vector of 32-bit float values of length 16 |
| `src1`  | fixed-length vector of 32-bit float values of length 16 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 6 |

### `rocdl.cvt.scalef32.f16.bf8`(ROCDL::CvtScaleF32F16Bf8Op)

*将bf8从打包向量缩放转换为f16，更新绑定结果*

语法：

```
operation ::= `rocdl.cvt.scalef32.f16.bf8` attr-dict $src `[` $srcSelIndex `]` `,` $scale `->` $oldVdst `[` $dstLoHiSel `]` `:` type($res)
```

将`src`中由`srcSelIndex`选取的bf8字节转换为f16，同时乘以`scale`的指数，并将结果存入`oldVdst`的第`dstLoHiSel`位，同时在返回值中保留该向量的其他元素。

字节以`i32`形式存储，而非`<4 x i8>`。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `srcSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `dstLoHiSel`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

|  Operand  | Description                                            |
| :-------: | ------------------------------------------------------ |
| `oldVdst` | fixed-length vector of 16-bit float values of length 2 |
|   `src`   | 32-bit signless integer                                |
|  `scale`  | 32-bit float                                           |

#### 结果：

| Result | Description                                            |
| :----: | ------------------------------------------------------ |
| `res`  | fixed-length vector of 16-bit float values of length 2 |

### `rocdl.cvt.scalef32.f16.fp8`(ROCDL::CvtScaleF32F16Fp8Op)

*将fp8从打包向量缩放转换为f16，并更新绑定结果*

语法：

```
operation ::= `rocdl.cvt.scalef32.f16.fp8` attr-dict $src `[` $srcSelIndex `]` `,` $scale `->` $oldVdst `[` $dstLoHiSel `]` `:` type($res)
```

将`src`中由`srcSelIndex`选取的fp8字节转换为f16，同时乘以`scale`的指数，并将结果存入`oldVdst`的第`dstLoHiSel`位，同时在返回值中保留该向量的其他元素。

字节以`i32`形式存储，而非`<4 x i8>`。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `srcSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `dstLoHiSel`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

|  Operand  | Description                                            |
| :-------: | ------------------------------------------------------ |
| `oldVdst` | fixed-length vector of 16-bit float values of length 2 |
|   `src`   | 32-bit signless integer                                |
|  `scale`  | 32-bit float                                           |

#### 结果：

| Result | Description                                            |
| :----: | ------------------------------------------------------ |
| `res`  | fixed-length vector of 16-bit float values of length 2 |

### `rocdl.cvt.scalef32.f32.bf8`(ROCDL::CvtScaleF32F32Bf8Op)

*将bf8从打包向量缩放转换为f32*

语法：

```
operation ::= `rocdl.cvt.scalef32.f32.bf8` attr-dict $src `[` $srcSelIndex `]` `,` $scale `:` type($res)
```

将`src`中由`srcSelIndex`选定的bf8字节转换为f32，并乘以`scale`的指数。

字节存储于`i32`类型中，而非`<4 x i8>`。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `srcSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `src`  | 32-bit signless integer |
| `scale` | 32-bit float            |

#### 结果：

| Result | Description  |
| :----: | ------------ |
| `res`  | 32-bit float |

### `rocdl.cvt.scalef32.f32.fp8`(ROCDL::CvtScaleF32F32Fp8Op)

*将 fp8 从打包向量缩放转换为 f32*

语法：

```
operation ::= `rocdl.cvt.scalef32.f32.fp8` attr-dict $src `[` $srcSelIndex `]` `,` $scale `:` type($res)
```

将`src`中由`srcSelIndex`选定的fp8字节转换为f32，并乘以`scale`的指数。

字节存储于`i32`类型中，而非`<4 x i8>`。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `srcSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `src`  | 32-bit signless integer |
| `scale` | 32-bit float            |

#### 结果：

| Result | Description  |
| :----: | ------------ |
| `res`  | 32-bit float |

### `rocdl.cvt.scalef32.pk.bf16.bf8`(ROCDL::CvtScaleF32PkBf16Bf8Op)

*将两个 BF8 缩放转换为两个 BF16*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.bf16.bf8` attr-dict $src `[` $srcLoHiSel `]` `,` $scale `:` type($res)
```

将`src0`中两个打包的bf8值转换为两个bf16值，并乘以`scale`中的指数。待转换的两个值根据`srcLoHiSel`从`src`（以`i32`表示的打包向量）的低半或高半中选取。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute    | MLIR Type           | Description                      |
| ------------ | ------------------- | -------------------------------- |
| `srcLoHiSel` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `src`  | 32-bit signless integer |
| `scale` | 32-bit float            |

#### 结果：

| Result | Description                                             |
| :----: | ------------------------------------------------------- |
| `res`  | fixed-length vector of bfloat16 type values of length 2 |

### `rocdl.cvt.scalef32.pk.bf16.fp4`(ROCDL::CvtScaleF32PkBf16Fp4Op)

*将两个打包的fp4缩放转换为打包的bf16*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.bf16.fp4` attr-dict $src `[` $srcSelIndex `]` `,` $scale `:` type($res)
```

将两个以32位整数单字节存储的打包fp4（f4E2M1）值转换为打包bf16，转换前乘以`scale`中的指数部分。

待转换的字节由`srcSelIndex`选择。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `srcSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `src`  | 32-bit signless integer |
| `scale` | 32-bit float            |

#### 结果：

| Result | Description                                             |
| :----: | ------------------------------------------------------- |
| `res`  | fixed-length vector of bfloat16 type values of length 2 |

### `rocdl.cvt.scalef32.pk.bf16.fp8`(ROCDL::CvtScaleF32PkBf16Fp8Op)

*将两个fp8缩放转换为两个bf16*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.bf16.fp8` attr-dict $src `[` $srcLoHiSel `]` `,` $scale `:` type($res)
```

将`src0`中两个打包fp8值转换为两个bf16值，并乘以`scale`中的指数。待转换的两个值根据`srcLoHiSel`从`src`（以`i32`表示的打包向量）的低半或高半中选取。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute    | MLIR Type           | Description                      |
| ------------ | ------------------- | -------------------------------- |
| `srcLoHiSel` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `src`  | 32-bit signless integer |
| `scale` | 32-bit float            |

#### 结果：

| Result | Description                                             |
| :----: | ------------------------------------------------------- |
| `res`  | fixed-length vector of bfloat16 type values of length 2 |

### `rocdl.cvt.scalef32.pk.bf8.bf16`(ROCDL::CvtScaleF32PkBf8Bf16Op)

*将两个bf16缩放转换为两个bf8，更新打包向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.bf8.bf16` attr-dict $src0 `,` $scale `->` $oldVdst `[` $dstLoHiSel `]` `:` type($res)
```

将`src0`中的两个bf16值转换为两个bf8字节，除以`scale`中的指数。字节被打包为16位值，插入到`oldVdst`的`dstLoHiSel`位置，并返回整个更新后的向量。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute    | MLIR Type           | Description                      |
| ------------ | ------------------- | -------------------------------- |
| `dstLoHiSel` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `oldVdst` | fixed-length vector of 16-bit signless integer values of length 2 |
|  `src0`   | fixed-length vector of bfloat16 type values of length 2      |
|  `scale`  | 32-bit float                                                 |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 16-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.pk.bf8.f16`(ROCDL::CvtScaleF32PkBf8F16Op)

*将两个f16缩放转换为两个bf8，更新打包向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.bf8.f16` attr-dict $src0 `,` $scale `->` $oldVdst `[` $dstLoHiSel `]` `:` type($res)
```

将`src0`中的两个f16值转换为两个bf8字节，除以`scale`中的指数。这些字节被打包成16位值，插入到`oldVdst`的`dstLoHiSel`位置，并返回整个更新后的向量。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute    | MLIR Type           | Description                      |
| ------------ | ------------------- | -------------------------------- |
| `dstLoHiSel` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `oldVdst` | fixed-length vector of 16-bit signless integer values of length 2 |
|  `src0`   | fixed-length vector of 16-bit float values of length 2       |
|  `scale`  | 32-bit float                                                 |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 16-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.pk.bf8.f32`(ROCDL::CvtScaleF32PkBf8F32Op)

*将两个f32缩放转换为两个bf8，更新打包向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.bf8.f32` attr-dict  $src0 `,` $src1 `,` $scale `->` $oldVdst `[` $dstLoHiSel `]` `:` type($res)
```

将`src0`和`src1`中的两个f32值转换为两个bf8字节，除以`scale`中的指数。字节被打包为16位值，插入到`oldVdst`的`dstLoHiSel`位置，并返回整个更新后的向量。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute    | MLIR Type           | Description                      |
| ------------ | ------------------- | -------------------------------- |
| `dstLoHiSel` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `oldVdst` | fixed-length vector of 16-bit signless integer values of length 2 |
|  `src0`   | 32-bit float                                                 |
|  `src1`   | 32-bit float                                                 |
|  `scale`  | 32-bit float                                                 |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 16-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.pk.f16.bf8`(ROCDL::CvtScaleF32PkF16Bf8Op)

*将两个bf8缩放转换为两个f16*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.f16.bf8` attr-dict $src `[` $srcLoHiSel `]` `,` $scale `:` type($res)
```

将`src0`中两个打包bf8值转换为两个f16值，乘以`scale`中的指数。待转换的两个值根据`srcLoHiSel`从`src`（以`i32`表示的打包向量）的低半或高半中选取。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute    | MLIR Type           | Description                      |
| ------------ | ------------------- | -------------------------------- |
| `srcLoHiSel` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `src`  | 32-bit signless integer |
| `scale` | 32-bit float            |

#### 结果：

| Result | Description                                            |
| :----: | ------------------------------------------------------ |
| `res`  | fixed-length vector of 16-bit float values of length 2 |

### `rocdl.cvt.scalef32.pk.f16.fp4`(ROCDL::CvtScaleF32PkF16Fp4Op)

*将两个打包 fp4 缩放转换为打包 f16*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.f16.fp4` attr-dict $src `[` $srcSelIndex `]` `,` $scale `:` type($res)
```

将两个以32位整数单字节存储的打包fp4（f4E2M1）值转换为打包f16，转换前乘以`scale`的指数部分。

待转换的字节由`srcSelIndex`选择。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `srcSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `src`  | 32-bit signless integer |
| `scale` | 32-bit float            |

#### 结果：

| Result | Description                                            |
| :----: | ------------------------------------------------------ |
| `res`  | fixed-length vector of 16-bit float values of length 2 |

### `rocdl.cvt.scalef32.pk.f16.fp8`(ROCDL::CvtScaleF32PkF16Fp8Op)

*将两个 fp8 缩放转换为两个 f16*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.f16.fp8` attr-dict $src `[` $srcLoHiSel `]` `,` $scale `:` type($res)
```

将`src0`中两个打包fp8值转换为两个f16值，乘以`scale`中的指数。待转换的两个值将根据`srcLoHiSel`，从`src`（以`i32`表示的打包向量）的低半部或高半部中选取。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute    | MLIR Type           | Description                      |
| ------------ | ------------------- | -------------------------------- |
| `srcLoHiSel` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `src`  | 32-bit signless integer |
| `scale` | 32-bit float            |

#### 结果：

| Result | Description                                            |
| :----: | ------------------------------------------------------ |
| `res`  | fixed-length vector of 16-bit float values of length 2 |

### `rocdl.cvt.scalef32.pk.f32.bf8`(ROCDL::CvtScaleF32PkF32Bf8Op)

*将两个 BF8 缩放转换为两个 F32*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.f32.bf8` attr-dict $src `[` $srcLoHiSel `]` `,` $scale `:` type($res)
```

将`src0`中两个打包bf8值转换为两个f32值，乘以`scale`中的指数。待转换的两个值将根据`srcLoHiSel`，从`src`（以`i32`表示的打包向量）的低半或高半中选取。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute    | MLIR Type           | Description                      |
| ------------ | ------------------- | -------------------------------- |
| `srcLoHiSel` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `src`  | 32-bit signless integer |
| `scale` | 32-bit float            |

#### 结果：

| Result | Description                                            |
| :----: | ------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit float values of length 2 |

### `rocdl.cvt.scalef32.pk.f32.fp4`(ROCDL::CvtScaleF32PkF32Fp4Op)

*将两个打包的 FP4 缩放转换为打包的 F32*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.f32.fp4` attr-dict $src `[` $srcSelIndex `]` `,` $scale `:` type($res)
```

将两个以32位整数单字节存储的打包fp4（f4E2M1）值转换为打包f32，转换前乘以`scale`的指数部分。

待转换的字节由`srcSelIndex`选择。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `srcSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `src`  | 32-bit signless integer |
| `scale` | 32-bit float            |

#### 结果：

| Result | Description                                            |
| :----: | ------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit float values of length 2 |

### `rocdl.cvt.scalef32.pk.f32.fp8`(ROCDL::CvtScaleF32PkF32Fp8Op)

*将两个fp8缩放转换为两个f32*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.f32.fp8` attr-dict $src `[` $srcLoHiSel `]` `,` $scale `:` type($res)
```

将`src0`中两个打包fp8值转换为两个f32值，乘以`scale`中的指数。待转换的两个值根据`srcLoHiSel`从`src`（以`i32`表示的打包向量）的低半或高半中选取。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute    | MLIR Type           | Description                      |
| ------------ | ------------------- | -------------------------------- |
| `srcLoHiSel` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `src`  | 32-bit signless integer |
| `scale` | 32-bit float            |

#### 结果：

| Result | Description                                            |
| :----: | ------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit float values of length 2 |

### `rocdl.cvt.scalef32.pk.fp4.bf16`(ROCDL::CvtScaleF32PkFp4Bf16Op)

*将两个bf16缩放转换为打包fp4，更新绑定向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.fp4.bf16` attr-dict $src `,` $scale `->` $oldVdst `[` $dstSelIndex `]` `:` type($res)
```

将两个打包bf16值转换为打包fp4，转换前先除以`scale`的指数部分。

两个缩放后的值被打包到一个字节中。该字节用于更新`oldVdst`的第`dstSelIndex`字节，最终整个oldVdst被返回。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `dstSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

|  Operand  | Description                                             |
| :-------: | ------------------------------------------------------- |
| `oldVdst` | 32-bit signless integer                                 |
|   `src`   | fixed-length vector of bfloat16 type values of length 2 |
|  `scale`  | 32-bit float                                            |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.pk.fp4.f16`(ROCDL::CvtScaleF32PkFp4F16Op)

*将两个f16缩放转换为打包fp4，更新绑定向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.fp4.f16` attr-dict $src `,` $scale `->` $oldVdst `[` $dstSelIndex `]` `:` type($res)
```

将两个打包 fp16 值转换为打包 fp4，转换前先除以`scale`的指数部分。

两个缩放后的值被打包到一个字节中。该字节用于更新`oldVdst`的第`dstSelIndex`字节，并返回完整的 oldVdst。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `dstSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

|  Operand  | Description                                            |
| :-------: | ------------------------------------------------------ |
| `oldVdst` | 32-bit signless integer                                |
|   `src`   | fixed-length vector of 16-bit float values of length 2 |
|  `scale`  | 32-bit float                                           |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.pk.fp4.f32`(ROCDL::CvtScaleF32PkFp4F32Op)

*将两个 f32 值缩放转换为两个打包的 fp4 值，更新绑定向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.fp4.f32` attr-dict $src0 `,` $src1 `,` $scale `->` $oldVdst `[` $dstSelIndex `]` `:` type($res)
```

将通过`src0`和`src1`传递的两个单精度浮点数转换为两个fp4值，转换前先将它们除以`scale`的指数部分。

两个缩放后的值被打包至一个字节。该字节用于更新`oldVdst`的第`dstSelIndex`位字节，最终返回完整oldVdst。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `dstSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `oldVdst` | 32-bit signless integer |
|  `src0`   | 32-bit float            |
|  `src1`   | 32-bit float            |
|  `scale`  | 32-bit float            |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.pk.fp8.bf16`(ROCDL::CvtScaleF32PkFp8Bf16Op)

*将两个bf16缩放转换为两个fp8，更新打包向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.fp8.bf16` attr-dict $src0 `,` $scale `->` $oldVdst `[` $dstLoHiSel `]` `:` type($res)
```

将`src0`中的两个bf16值转换为两个fp8字节，除以`scale`中的指数。这些字节被打包成16位值，插入到`oldVdst`的`dstLoHiSel`位置，并返回整个更新后的向量。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute    | MLIR Type           | Description                      |
| ------------ | ------------------- | -------------------------------- |
| `dstLoHiSel` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `oldVdst` | fixed-length vector of 16-bit signless integer values of length 2 |
|  `src0`   | fixed-length vector of bfloat16 type values of length 2      |
|  `scale`  | 32-bit float                                                 |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 16-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.pk.fp8.f16`(ROCDL::CvtScaleF32PkFp8F16Op)

*将两个f16缩放转换为两个fp8，更新打包向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.fp8.f16` attr-dict $src0 `,` $scale `->` $oldVdst `[` $dstLoHiSel `]` `:` type($res)
```

将`src0`中的两个f16值转换为两个fp8字节，除以`scale`中的指数。字节被打包为16位值，插入到`oldVdst`的`dstLoHiSel`位置，并返回整个更新后的向量。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute    | MLIR Type           | Description                      |
| ------------ | ------------------- | -------------------------------- |
| `dstLoHiSel` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `oldVdst` | fixed-length vector of 16-bit signless integer values of length 2 |
|  `src0`   | fixed-length vector of 16-bit float values of length 2       |
|  `scale`  | 32-bit float                                                 |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 16-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.pk.fp8.f32`(ROCDL::CvtScaleF32PkFp8F32Op)

*将两个 f32 缩放转换为两个 fp8，更新打包向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk.fp8.f32` attr-dict  $src0 `,` $src1 `,` $scale `->` $oldVdst `[` $dstLoHiSel `]` `:` type($res)
```

将`src0`和`src1`中的两个f32值转换为两个fp8字节，除以`scale`中的指数。字节被打包为16位值，插入到`oldVdst`的`dstLoHiSel`位置，并返回整个更新后的向量。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute    | MLIR Type           | Description                      |
| ------------ | ------------------- | -------------------------------- |
| `dstLoHiSel` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `oldVdst` | fixed-length vector of 16-bit signless integer values of length 2 |
|  `src0`   | 32-bit float                                                 |
|  `src1`   | 32-bit float                                                 |
|  `scale`  | 32-bit float                                                 |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 16-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.pk16.bf6.bf16`(ROCDL::CvtScaleF32Pk16Bf6Bf16Op)

*将打包bf16缩放转换为打包bf6*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk16.bf6.bf16` attr-dict $src `,` $scale `:` type($res)
```

将8个打包bf16值转换为打包bf6，转换前乘以`scale`的指数部分。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | fixed-length vector of bfloat16 type values of length 16 |
| `scale` | 32-bit float                                             |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 3 |

### `rocdl.cvt.scalef32.pk16.bf6.f16`(ROCDL::CvtScaleF32Pk16Bf6F16Op)

*将打包的f16缩放转换为打包的bf6*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk16.bf6.f16` attr-dict $src `,` $scale `:` type($res)
```

将8个打包f16值转换为打包bf6，转换前乘以`scale`的指数部分。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of 16-bit float values of length 16 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 3 |

### `rocdl.cvt.scalef32.pk16.bf6.f32`(ROCDL::CvtScaleF32Pk16Bf6F32Op)

*将打包f32缩放转换为打包bf6*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk16.bf6.f32` attr-dict $src `,` $scale `:` type($res)
```

将8个打包f32值转换为打包bf6，转换前乘以缩放系数的`scale`部分。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of 32-bit float values of length 16 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 3 |

### `rocdl.cvt.scalef32.pk16.fp6.bf16`(ROCDL::CvtScaleF32Pk16Fp6Bf16Op)

*将打包的bf16缩放转换为打包的fp6*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk16.fp6.bf16` attr-dict $src `,` $scale `:` type($res)
```

将8个打包bf16值转换为打包fp6，转换前乘以`scale`的指数部分。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | fixed-length vector of bfloat16 type values of length 16 |
| `scale` | 32-bit float                                             |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 3 |

### `rocdl.cvt.scalef32.pk16.fp6.f16`(ROCDL::CvtScaleF32Pk16Fp6F16Op)

*将打包f16缩放转换为打包fp6*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk16.fp6.f16` attr-dict $src `,` $scale `:` type($res)
```

将8个打包f16值转换为打包fp6，转换前乘以`scale`的指数部分。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of 16-bit float values of length 16 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 3 |

### `rocdl.cvt.scalef32.pk16.fp6.f32`(ROCDL::CvtScaleF32Pk16Fp6F32Op)

*将打包f32缩放转换为打包fp6*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk16.fp6.f32` attr-dict $src `,` $scale `:` type($res)
```

将8个打包f32值转换为打包fp6，转换前乘以`scale`的指数部分。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of 32-bit float values of length 16 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 3 |

### `rocdl.cvt.scalef32.pk32.bf16.bf6`(ROCDL::CvtScaleF32Pk32Bf16Bf6Op)

*将打包bf6缩放转换为打包bf16*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk32.bf16.bf6` attr-dict $src `,` $scale `:` type($res)
```

将32个打包bf6值转换为打包bf16，转换前乘以`scale`的指数部分。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 6 |
| `scale` | 32-bit float                                                 |

#### 结果：

| Result | Description                                              |
| :----: | -------------------------------------------------------- |
| `res`  | fixed-length vector of bfloat16 type values of length 32 |

### `rocdl.cvt.scalef32.pk32.bf16.fp6`(ROCDL::CvtScaleF32Pk32Bf16Fp6Op)

*将打包的fp6缩放转换为打包的bf16*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk32.bf16.fp6` attr-dict $src `,` $scale `:` type($res)
```

将32个打包fp6值转换为打包bf16，转换前乘以`scale`的指数部分。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 6 |
| `scale` | 32-bit float                                                 |

#### 结果：

| Result | Description                                              |
| :----: | -------------------------------------------------------- |
| `res`  | fixed-length vector of bfloat16 type values of length 32 |

### `rocdl.cvt.scalef32.pk32.bf6.bf16`(ROCDL::CvtScaleF32Pk32Bf6Bf16Op)

*将打包的bf16缩放转换为打包bf6*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk32.bf6.bf16` attr-dict $src `,` $scale `:` type($res)
```

将32个打包bf16值转换为打包bf6，转换前先除以`scale`的指数部分。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | fixed-length vector of bfloat16 type values of length 32 |
| `scale` | 32-bit float                                             |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 6 |

### `rocdl.cvt.scalef32.pk32.bf6.f16`(ROCDL::CvtScaleF32Pk32Bf6F16Op)

*将打包f16缩放转换为打包bf6*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk32.bf6.f16` attr-dict $src `,` $scale `:` type($res)
```

将32个打包f16值转换为打包bf6，转换前需除以`scale`的指数部分。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of 16-bit float values of length 32 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 6 |

### `rocdl.cvt.scalef32.pk32.f16.bf6`(ROCDL::CvtScaleF32Pk32F16Bf6Op)

*将打包bf6缩放转换为打包f16*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk32.f16.bf6` attr-dict $src `,` $scale `:` type($res)
```

将32个打包bf6值转换为打包f16，转换前乘以`scale`的指数部分。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 6 |
| `scale` | 32-bit float                                                 |

#### 结果：

| Result | Description                                             |
| :----: | ------------------------------------------------------- |
| `res`  | fixed-length vector of 16-bit float values of length 32 |

### `rocdl.cvt.scalef32.pk32.f16.fp6`(ROCDL::CvtScaleF32Pk32F16Fp6Op)

*将打包fp6缩放转换为打包f16*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk32.f16.fp6` attr-dict $src `,` $scale `:` type($res)
```

将32个打包fp6值转换为打包f16，转换前乘以`scale`的指数部分。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 6 |
| `scale` | 32-bit float                                                 |

#### 结果：

| Result | Description                                             |
| :----: | ------------------------------------------------------- |
| `res`  | fixed-length vector of 16-bit float values of length 32 |

### `rocdl.cvt.scalef32.pk32.f32.bf6`(ROCDL::CvtScaleF32Pk32F32Bf6Op)

*将打包的bf6缩放转换为打包的f32*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk32.f32.bf6` attr-dict $src `,` $scale `:` type($res)
```

将32个打包bf6值转换为打包f32，转换前乘以`scale`的指数部分。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 6 |
| `scale` | 32-bit float                                                 |

#### 结果：

| Result | Description                                             |
| :----: | ------------------------------------------------------- |
| `res`  | fixed-length vector of 32-bit float values of length 32 |

### `rocdl.cvt.scalef32.pk32.f32.fp6`(ROCDL::CvtScaleF32Pk32F32Fp6Op)

*将打包fp6缩放转换为打包f32*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk32.f32.fp6` attr-dict $src `,` $scale `:` type($res)
```

将32个打包fp6值转换为打包f32，转换前乘以`scale`的指数部分。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit signless integer values of length 6 |
| `scale` | 32-bit float                                                 |

#### 结果：

| Result | Description                                             |
| :----: | ------------------------------------------------------- |
| `res`  | fixed-length vector of 32-bit float values of length 32 |

### `rocdl.cvt.scalef32.pk32.fp6.bf16`(ROCDL::CvtScaleF32Pk32Fp6Bf16Op)

*将打包bf16缩放转换为打包fp6*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk32.fp6.bf16` attr-dict $src `,` $scale `:` type($res)
```

将32个打包bf16值转换为打包fp6，转换前需除以`scale`的指数部分。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | fixed-length vector of bfloat16 type values of length 32 |
| `scale` | 32-bit float                                             |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 6 |

### `rocdl.cvt.scalef32.pk32.fp6.f16`(ROCDL::CvtScaleF32Pk32Fp6F16Op)

*将打包f16缩放转换为打包fp6*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk32.fp6.f16` attr-dict $src `,` $scale `:` type($res)
```

将32个打包f16值转换为打包fp6，转换前先除以`scale`的指数部分。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of 16-bit float values of length 32 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 6 |

### `rocdl.cvt.scalef32.pk8.bf8.bf16`(ROCDL::CvtScaleF32Pk8Bf8Bf16Op)

*将打包bf16缩放转换为打包bf8*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk8.bf8.bf16` attr-dict $src `,` $scale `:` type($res)
```

将8个打包bf16值转换为打包bf8，转换前乘以`scale`的指数部分。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of bfloat16 type values of length 8 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.pk8.bf8.f16`(ROCDL::CvtScaleF32Pk8Bf8F16Op)

*将打包f16缩放转换为打包bf8*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk8.bf8.f16` attr-dict $src `,` $scale `:` type($res)
```

将8个打包f16值转换为打包bf8，转换前乘以`scale`的指数部分。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                            |
| :-----: | ------------------------------------------------------ |
|  `src`  | fixed-length vector of 16-bit float values of length 8 |
| `scale` | 32-bit float                                           |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.pk8.bf8.f32`(ROCDL::CvtScaleF32Pk8Bf8F32Op)

*将打包f32缩放转换为打包bf8*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk8.bf8.f32` attr-dict $src `,` $scale `:` type($res)
```

将8个打包f32值转换为打包bf8，转换前乘以`scale`的指数部分。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                            |
| :-----: | ------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit float values of length 8 |
| `scale` | 32-bit float                                           |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.pk8.fp4.bf16`(ROCDL::CvtScaleF32Pk8Fp4Bf16Op)

*将打包的bf16缩放转换为打包的fp4*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk8.fp4.bf16` attr-dict $src `,` $scale `:` type($res)
```

将8个打包bf16值转换为打包fp4，转换前乘以`scale`的指数部分。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of bfloat16 type values of length 8 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.pk8.fp4.f16`(ROCDL::CvtScaleF32Pk8Fp4F16Op)

*将打包f16缩放转换为打包fp4*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk8.fp4.f16` attr-dict $src `,` $scale `:` type($res)
```

将8个打包f16值转换为打包fp4，转换前乘以`scale`的指数部分。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                            |
| :-----: | ------------------------------------------------------ |
|  `src`  | fixed-length vector of 16-bit float values of length 8 |
| `scale` | 32-bit float                                           |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.pk8.fp4.f32`(ROCDL::CvtScaleF32Pk8Fp4F32Op)

*将打包f32缩放转换为打包fp4*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk8.fp4.f32` attr-dict $src `,` $scale `:` type($res)
```

将8个打包f32值转换为打包fp4，转换前需乘以`scale`的指数部分。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                            |
| :-----: | ------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit float values of length 8 |
| `scale` | 32-bit float                                           |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.pk8.fp8.bf16`(ROCDL::CvtScaleF32Pk8Fp8Bf16Op)

*将打包bf16缩放转换为打包fp8*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk8.fp8.bf16` attr-dict $src `,` $scale `:` type($res)
```

将8个打包bf16值转换为打包fp8，转换前乘以`scale`的指数部分。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of bfloat16 type values of length 8 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.pk8.fp8.f16`(ROCDL::CvtScaleF32Pk8Fp8F16Op)

*将打包f16缩放转换为打包fp8*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk8.fp8.f16` attr-dict $src `,` $scale `:` type($res)
```

将8个打包f16值转换为打包fp8，转换前乘以`scale`的指数部分。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                            |
| :-----: | ------------------------------------------------------ |
|  `src`  | fixed-length vector of 16-bit float values of length 8 |
| `scale` | 32-bit float                                           |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.pk8.fp8.f32`(ROCDL::CvtScaleF32Pk8Fp8F32Op)

*将打包f32缩放转换为打包fp8*

语法：

```
operation ::= `rocdl.cvt.scalef32.pk8.fp8.f32` attr-dict $src `,` $scale `:` type($res)
```

将8个打包f32值转换为打包fp8，转换前乘以`scale`的指数部分。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                            |
| :-----: | ------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit float values of length 8 |
| `scale` | 32-bit float                                           |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.sr.bf8.bf16`(ROCDL::CvtScaleF32SrBf8BF16Op)

*采用随机舍入将bf16缩放转换为bf8，更新打包向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.bf8.bf16` attr-dict $src0 `,` $seed `,` $scale `->` $oldVdst `[` $dstSelIndex `]` `:` type($res)
```

将`src0`中的bf16值转换为bf8字节，除以`scale`中的指数，并使用`seed`进行随机化舍入。将结果字节存入`oldVdst`的第`dstSelIndex`位，并返回整个打包向量（存储为`i32`类型）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `dstSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `oldVdst` | 32-bit signless integer |
|  `src0`   | bfloat16 type           |
|  `seed`   | 32-bit signless integer |
|  `scale`  | 32-bit float            |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.sr.bf8.f16`(ROCDL::CvtScaleF32SrBf8F16Op)

*Scaled convert f16to bf8 with stochiastic rounding, updating packed vector*采用随机舍入将f16缩放转换为bf8，更新打包向量

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.bf8.f16` attr-dict $src0 `,` $seed `,` $scale `->` $oldVdst `[` $dstSelIndex `]` `:` type($res)
```

将`src0`中的f16值转换为bf8字节，除以`scale`中的指数，并使用`seed`进行随机化舍入。将结果字节存入`oldVds`t的第`dstSelIndex`位，并返回整个打包向量（存储为`i32`类型）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `dstSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `oldVdst` | 32-bit signless integer |
|  `src0`   | 16-bit float            |
|  `seed`   | 32-bit signless integer |
|  `scale`  | 32-bit float            |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.sr.bf8.f32`(ROCDL::CvtScaleF32SrBf8F32Op)

*采用随机舍入将f32缩放转换为bf8，更新打包向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.bf8.f32` attr-dict $src0 `,` $seed `,` $scale `->` $oldVdst `[` $dstSelIndex `]` `:` type($res)
```

将`src0`中的f32值转换为bf8字节，除以`scale`中的指数并使用`seed`进行随机舍入。将结果字节存入`oldVdst`的第`dstSelIndex`位，并返回整个打包向量（以`i32`形式存储）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `dstSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `oldVdst` | 32-bit signless integer |
|  `src0`   | 32-bit float            |
|  `seed`   | 32-bit signless integer |
|  `scale`  | 32-bit float            |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.sr.fp8.bf16`(ROCDL::CvtScaleF32SrFp8BF16Op)

*使用随机舍入将bf16缩放转换为fp8，更新打包向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.fp8.bf16` attr-dict $src0 `,` $seed `,` $scale `->` $oldVdst `[` $dstSelIndex `]` `:` type($res)
```

将`src0`中的bf16值转换为fp8字节，除以`scale`中的指数，并使用`seed`进行随机化舍入。将结果字节存入`oldVdst`的第`dstSelIndex`位，并返回整个打包向量（存储为`i32`类型）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `dstSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `oldVdst` | 32-bit signless integer |
|  `src0`   | bfloat16 type           |
|  `seed`   | 32-bit signless integer |
|  `scale`  | 32-bit float            |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.sr.fp8.f16`(ROCDL::CvtScaleF32SrFp8F16Op)

*采用随机舍入将 f16 缩放转换为 fp8，更新打包向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.fp8.f16` attr-dict $src0 `,` $seed `,` $scale `->` $oldVdst `[` $dstSelIndex `]` `:` type($res)
```

将`src0`中的f16值转换为fp8字节，除以`scale`中的指数，并使用`seed`进行随机化舍入。将结果字节存入`oldVdst`的第`dstSelIndex`位，并返回整个打包向量（存储为`i32`类型）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `dstSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `oldVdst` | 32-bit signless integer |
|  `src0`   | 16-bit float            |
|  `seed`   | 32-bit signless integer |
|  `scale`  | 32-bit float            |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.sr.fp8.f32`(ROCDL::CvtScaleF32SrFp8F32Op)

*采用随机舍入将f32缩放转换为fp8，更新打包向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.fp8.f32` attr-dict $src0 `,` $seed `,` $scale `->` $oldVdst `[` $dstSelIndex `]` `:` type($res)
```

将`src0`中的f32值转换为fp8字节，除以`scale`中的指数并使用`seed`进行随机舍入。将结果字节存入`oldVdst`的第`dstSelIndex`位，并返回整个打包向量（以`i32`形式存储）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `dstSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `oldVdst` | 32-bit signless integer |
|  `src0`   | 32-bit float            |
|  `seed`   | 32-bit signless integer |
|  `scale`  | 32-bit float            |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.sr.pk.fp4.bf16`(ROCDL::CvtScaleF32SrPkFp4Bf16Op)

*将两个bf16缩放转换为打包fp4，采用随机舍入，更新绑定向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk.fp4.bf16` attr-dict $src `,` $seed `,` $scale `->` $oldVdst `[` $dstSelIndex `]` `:` type($res)
```

将两个打包bf16值转换为打包fp4，转换前先除以`scale`的指数部分，并使用`seed`作为随机舍入的随机种子。

两个缩放后的值以小端序打包至一个字节。该字节用于更新`oldVdst`中第`dstSelIndex`位的字节，最终返回完整目标向量。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `dstSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

|  Operand  | Description                                             |
| :-------: | ------------------------------------------------------- |
| `oldVdst` | 32-bit signless integer                                 |
|   `src`   | fixed-length vector of bfloat16 type values of length 2 |
|  `seed`   | 32-bit signless integer                                 |
|  `scale`  | 32-bit float                                            |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.sr.pk.fp4.f16`(ROCDL::CvtScaleF32SrPkFp4F16Op)

*将两个 f16 缩放转换为打包 fp4，采用随机舍入，更新绑定向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk.fp4.f16` attr-dict $src `,` $seed `,` $scale `->` $oldVdst `[` $dstSelIndex `]` `:` type($res)
```

将两个打包 f16 值转换为打包 fp4，转换前先除以`scale`的指数部分，并使用`seed`作为随机舍入的随机种子。

两个缩放后的值以小端序打包为一个字节。该字节用于更新`oldVdst`的第`dstSelIndex`位字节，最终返回完整 oldVdst。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `dstSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

|  Operand  | Description                                            |
| :-------: | ------------------------------------------------------ |
| `oldVdst` | 32-bit signless integer                                |
|   `src`   | fixed-length vector of 16-bit float values of length 2 |
|  `seed`   | 32-bit signless integer                                |
|  `scale`  | 32-bit float                                           |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.sr.pk.fp4.f32`(ROCDL::CvtScaleF32SrPkFp4F32Op)

*将两个 f32 缩放转换为打包 fp4，采用随机舍入，更新绑定向量*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk.fp4.f32` attr-dict $src `,` $seed `,` $scale `->` $oldVdst `[` $dstSelIndex `]` `:` type($res)
```

将两个打包 f32 值转换为打包 fp4，转换前先除以`scale`的指数部分，并使用`seed`作为随机舍入的随机种子。

两个缩放后的值以小端序打包为一个字节。该字节用于更新`oldVdst`的`dstSelIndex`位字节，最终返回整个 oldVdst 值。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `dstSelIndex` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

|  Operand  | Description                                            |
| :-------: | ------------------------------------------------------ |
| `oldVdst` | 32-bit signless integer                                |
|   `src`   | fixed-length vector of 32-bit float values of length 2 |
|  `seed`   | 32-bit signless integer                                |
|  `scale`  | 32-bit float                                           |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.sr.pk16.bf6.bf16`(ROCDL::CvtScaleF32SrPk16Bf6Bf16Op)

*将打包bf16缩放转换为打包bf6，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk16.bf6.bf16` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将8个打包bf16值转换为打包bf6，转换前乘以`scale`的指数部分并应用随机化舍入。本操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | fixed-length vector of bfloat16 type values of length 16 |
| `seed`  | 32-bit signless integer                                  |
| `scale` | 32-bit float                                             |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 3 |

### `rocdl.cvt.scalef32.sr.pk16.bf6.f16`(ROCDL::CvtScaleF32SrPk16Bf6F16Op)

*将打包 f16 缩放转换为打包 bf6，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk16.bf6.f16` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将8个打包f16值转换为打包bf6，转换前乘以`scale`的指数部分并应用随机化舍入。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of 16-bit float values of length 16 |
| `seed`  | 32-bit signless integer                                 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 3 |

### `rocdl.cvt.scalef32.sr.pk16.bf6.f32`(ROCDL::CvtScaleF32SrPk16Bf6F32Op)

*将打包f32缩放转换为打包bf6，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk16.bf6.f32` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将8个打包f32值转换为打包bf6，转换前乘以`scale`中的指数部分并应用随机化舍入。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of 32-bit float values of length 16 |
| `seed`  | 32-bit signless integer                                 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 3 |

### `rocdl.cvt.scalef32.sr.pk16.fp6.bf16`(ROCDL::CvtScaleF32SrPk16Fp6Bf16Op)

*将打包bf16缩放转换为打包fp6，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk16.fp6.bf16` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将8个打包bf16值转换为打包fp6，转换前乘以`scale`的指数部分并应用随机化舍入。本操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | fixed-length vector of bfloat16 type values of length 16 |
| `seed`  | 32-bit signless integer                                  |
| `scale` | 32-bit float                                             |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 3 |

### `rocdl.cvt.scalef32.sr.pk16.fp6.f16`(ROCDL::CvtScaleF32SrPk16Fp6F16Op)

*将打包f16缩放转换为打包fp6，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk16.fp6.f16` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将8个打包f16值转换为打包fp6，转换前乘以`scale`中的指数部分并应用随机化舍入。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of 16-bit float values of length 16 |
| `seed`  | 32-bit signless integer                                 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 3 |

### `rocdl.cvt.scalef32.sr.pk16.fp6.f32`(ROCDL::CvtScaleF32SrPk16Fp6F32Op)

*将打包 f32 缩放转换为打包 fp6，使用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk16.fp6.f32` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将8个打包f32值转换为打包fp6，转换前乘以`scale`中的指数部分并应用随机化舍入。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of 32-bit float values of length 16 |
| `seed`  | 32-bit signless integer                                 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 3 |

### `rocdl.cvt.scalef32.sr.pk32.bf6.bf16`(ROCDL::CvtScaleF32SrPk32Bf6Bf16Op)

*将打包bf16缩放转换为打包bf6，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk32.bf6.bf16` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将32个打包bf16值转换为打包bf6，转换前先除以`scale`的指数部分，并应用基于`seed`生成的随机舍入。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | fixed-length vector of bfloat16 type values of length 32 |
| `seed`  | 32-bit signless integer                                  |
| `scale` | 32-bit float                                             |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 6 |

### `rocdl.cvt.scalef32.sr.pk32.bf6.f16`(ROCDL::CvtScaleF32SrPk32Bf6F16Op)

*将打包 f16 缩放转换为打包 bf6，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk32.bf6.f16` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将32个打包f16值转换为打包bf6，转换前先除以`scale`的指数部分，并应用基于`seed`生成的随机舍入。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of 16-bit float values of length 32 |
| `seed`  | 32-bit signless integer                                 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 6 |

### `rocdl.cvt.scalef32.sr.pk32.bf6.f32`(ROCDL::CvtScaleF32SrPk32Bf6F32Op)

*将打包f32缩放转换为打包bf6，采用随机化舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk32.bf6.f32` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将32个打包f32值转换为打包bf6，转换前先除以`scale`的指数部分，并应用基于`seed`生成的随机舍入。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of 32-bit float values of length 32 |
| `seed`  | 32-bit signless integer                                 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 6 |

### `rocdl.cvt.scalef32.sr.pk32.fp6.bf16`(ROCDL::CvtScaleF32SrPk32Fp6Bf16Op)

*将打包bf16缩放转换为打包fp6，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk32.fp6.bf16` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将32个打包bf16值转换为打包fp6，转换前先除以`scale`的指数部分，并应用基于`seed`生成的随机舍入。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | fixed-length vector of bfloat16 type values of length 32 |
| `seed`  | 32-bit signless integer                                  |
| `scale` | 32-bit float                                             |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 6 |

### `rocdl.cvt.scalef32.sr.pk32.fp6.f16`(ROCDL::CvtScaleF32SrPk32Fp6F16Op)

*将打包f16缩放转换为打包fp6，采用随机化舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk32.fp6.f16` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将32个打包f16值转换为打包fp6，转换前先除以`scale`中的指数部分，并应用基于`seed`生成的随机舍入。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of 16-bit float values of length 32 |
| `seed`  | 32-bit signless integer                                 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 6 |

### `rocdl.cvt.scalef32.sr.pk32.fp6.f32`(ROCDL::CvtScaleF32SrPk32Fp6F32Op)

*将打包 f32 缩放转换为打包 fp6，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk32.fp6.f32` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将32个打包f32值转换为打包fp6，转换前先除以`scale`的指数部分，并应用基于`seed`生成的随机舍入。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of 32-bit float values of length 32 |
| `seed`  | 32-bit signless integer                                 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 6 |

### `rocdl.cvt.scalef32.sr.pk8.bf8.bf16`(ROCDL::CvtScaleF32SrPk8Bf8Bf16Op)

*将打包bf16缩放转换为打包bf8，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk8.bf8.bf16` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将8个打包bf16值转换为打包bf8，转换前乘以`scale`的指数部分并应用随机化舍入。本操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of bfloat16 type values of length 8 |
| `seed`  | 32-bit signless integer                                 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.sr.pk8.bf8.f16`(ROCDL::CvtScaleF32SrPk8Bf8F16Op)

*将打包 f16 缩放转换为打包 bf8，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk8.bf8.f16` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将8个打包f16值转换为打包bf8，转换前乘以`scale`的指数部分并应用随机化舍入。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                            |
| :-----: | ------------------------------------------------------ |
|  `src`  | fixed-length vector of 16-bit float values of length 8 |
| `seed`  | 32-bit signless integer                                |
| `scale` | 32-bit float                                           |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.sr.pk8.bf8.f32`(ROCDL::CvtScaleF32SrPk8Bf8F32Op)

*将打包f32缩放转换为打包bf8，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk8.bf8.f32` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将8个打包f32值转换为打包bf8，转换前乘以`scale`的指数部分并应用随机化舍入。本操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                            |
| :-----: | ------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit float values of length 8 |
| `seed`  | 32-bit signless integer                                |
| `scale` | 32-bit float                                           |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.sr.pk8.fp4.bf16`(ROCDL::CvtScaleF32SrPk8Fp4Bf16Op)

*将打包bf16缩放转换为打包fp4，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk8.fp4.bf16` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将8个打包bf16值转换为打包fp4，转换前乘以`scale`的指数部分，并应用随机舍入。本操作仅适用于 gfx1250+ 架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of bfloat16 type values of length 8 |
| `seed`  | 32-bit signless integer                                 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.sr.pk8.fp4.f16`(ROCDL::CvtScaleF32SrPk8Fp4F16Op)

*将打包f16缩放转换为打包fp4，并采用随机化舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk8.fp4.f16` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将8个打包f16值转换为打包fp4，转换前乘以`scale`的指数部分并应用随机化舍入。本操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                            |
| :-----: | ------------------------------------------------------ |
|  `src`  | fixed-length vector of 16-bit float values of length 8 |
| `seed`  | 32-bit signless integer                                |
| `scale` | 32-bit float                                           |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.sr.pk8.fp4.f32`(ROCDL::CvtScaleF32SrPk8Fp4F32Op)

*将打包 f32 缩放转换为打包 fp4，使用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk8.fp4.f32` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将8个打包f32值转换为打包fp4，转换前乘以`scale`的指数部分，并应用随机舍入。本操作仅适用于 gfx1250+ 架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                            |
| :-----: | ------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit float values of length 8 |
| `seed`  | 32-bit signless integer                                |
| `scale` | 32-bit float                                           |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.cvt.scalef32.sr.pk8.fp8.bf16`(ROCDL::CvtScaleF32SrPk8Fp8Bf16Op)

*将打包bf16缩放转换为打包fp8，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk8.fp8.bf16` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将8个打包bf16值转换为打包fp8，转换前乘以`scale`的指数部分并应用随机舍入。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                             |
| :-----: | ------------------------------------------------------- |
|  `src`  | fixed-length vector of bfloat16 type values of length 8 |
| `seed`  | 32-bit signless integer                                 |
| `scale` | 32-bit float                                            |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.sr.pk8.fp8.f16`(ROCDL::CvtScaleF32SrPk8Fp8F16Op)

*将打包 f16 缩放转换为打包 fp8，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk8.fp8.f16` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将8个打包f16值转换为打包fp8，转换前乘以`scale`的指数部分并应用随机化舍入。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                            |
| :-----: | ------------------------------------------------------ |
|  `src`  | fixed-length vector of 16-bit float values of length 8 |
| `seed`  | 32-bit signless integer                                |
| `scale` | 32-bit float                                           |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 2 |

### `rocdl.cvt.scalef32.sr.pk8.fp8.f32`(ROCDL::CvtScaleF32SrPk8Fp8F32Op)

*将打包f32缩放转换为打包fp8，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.scalef32.sr.pk8.fp8.f32` attr-dict $src `,` $seed `,` $scale `:` type($res)
```

将8个打包f32值转换为打包fp8，转换前乘以`scale`的指数部分并应用随机化舍入。此操作适用于gfx1250+架构。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                            |
| :-----: | ------------------------------------------------------ |
|  `src`  | fixed-length vector of 32-bit float values of length 8 |
| `seed`  | 32-bit signless integer                                |
| `scale` | 32-bit float                                           |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | fixed-length vector of 32-bit signless integer values of length 2 |

### `rocdl.cvt.sr.bf8.f32`(ROCDL::CvtSrBf8F32Op)

*将 f32 转换为 bf8，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.sr.bf8.f32` attr-dict $srcA `,` $srcB `->` $old `[` $byteSel `]` `:` type($res)
```

将`srcA`转换为 bf8，添加`srcB`中的舍入因子，并存储到`old`的第`byteSel`字节中，保留其余位。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `byteSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
| `srcA`  | 32-bit float            |
| `srcB`  | 32-bit signless integer |
|  `old`  | 32-bit signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.cvt.sr.fp8.f32`(ROCDL::CvtSrFp8F32Op)

*将 f32 转换为 fp8，采用随机舍入*

语法：

```
operation ::= `rocdl.cvt.sr.fp8.f32` attr-dict $srcA `,` $srcB `->` $old `[` $byteSel `]` `:` type($res)
```

将`srcA`转换为fp8，添加`srcB`中的舍入因子，并存储到`old`的第`byteSel`字节中，保留其余字节。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `byteSel` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
| `srcA`  | 32-bit float            |
| `srcB`  | 32-bit signless integer |
|  `old`  | 32-bit signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.ds.load.tr16.b128`(ROCDL::DsLoadTr16_B128)

*从ds内存加载并转置矩阵至寄存器（gfx1250及以上型号可用）。*

语法：

```
operation ::= `rocdl.ds.load.tr16.b128` $ptr attr-dict `:` qualified(type($ptr)) `->` type($res)
```

从ds内存加载16位数据矩阵，在行优先与列优先顺序间转置数据，并将结果存储至128位向量寄存器。

支持版本：gfx1250+。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
|  `ptr`  | LLVM pointer in address space 3 |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.ds.load.tr4.b64`(ROCDL::DsLoadTr4_B64)

*从ds内存加载并转置矩阵至寄存器（gfx1250及以上型号可用）。*

语法：

```
operation ::= `rocdl.ds.load.tr4.b64` $ptr attr-dict `:` qualified(type($ptr)) `->` type($res)
```

从ds内存加载4位数据矩阵，在行优先与列优先顺序间进行数据转置，并将结果存储至64位向量寄存器。

支持gfx1250及以上版本。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
|  `ptr`  | LLVM pointer in address space 3 |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.ds.load.tr6.b96`(ROCDL::DsLoadTr6_B96)

*从ds内存加载并转置矩阵至寄存器（gfx1250及以上型号可用）。*

语法：

```
operation ::= `rocdl.ds.load.tr6.b96` $ptr attr-dict `:` qualified(type($ptr)) `->` type($res)
```

从ds内存加载6位数据矩阵，在行优先与列优先顺序间转置数据，并将结果存储至96位向量寄存器。

支持版本：gfx1250+。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
|  `ptr`  | LLVM pointer in address space 3 |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.ds.load.tr8.b64`(ROCDL::DsLoadTr8_B64)

*从ds内存加载并转置矩阵至寄存器（gfx1250及以上型号可用）。*

语法：

```
operation ::= `rocdl.ds.load.tr8.b64` $ptr attr-dict `:` qualified(type($ptr)) `->` type($res)
```

从ds内存加载8位数据矩阵，在行优先与列优先顺序间进行数据转置，并将结果存储至64位向量寄存器。

支持gfx1250及以上版本。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
|  `ptr`  | LLVM pointer in address space 3 |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.ds.read.tr16.b64`(ROCDL::ds_read_tr16_b64)

语法：

```
operation ::= `rocdl.ds.read.tr16.b64` $ptr attr-dict `:` type($ptr) `->` type($res)
```

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
|  `ptr`  | LLVM pointer in address space 3 |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.ds.read.tr4.b64`(ROCDL::ds_read_tr4_b64)

语法：

```
operation ::= `rocdl.ds.read.tr4.b64` $ptr attr-dict `:` type($ptr) `->` type($res)
```

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
|  `ptr`  | LLVM pointer in address space 3 |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.ds.read.tr6.b96`(ROCDL::ds_read_tr6_b96)

语法：

```
operation ::= `rocdl.ds.read.tr6.b96` $ptr attr-dict `:` type($ptr) `->` type($res)
```

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
|  `ptr`  | LLVM pointer in address space 3 |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.ds.read.tr8.b64`(ROCDL::ds_read_tr8_b64)

语法：

```
operation ::= `rocdl.ds.read.tr8.b64` $ptr attr-dict `:` type($ptr) `->` type($res)
```

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
|  `ptr`  | LLVM pointer in address space 3 |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.ds_bpermute`(ROCDL::DsBpermuteOp)

语法：

```
operation ::= `rocdl.ds_bpermute` $index `,` $src  attr-dict `:` `(` type($index) `,` type($src) `)` `->` type($res)
```

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
| `index` | 32-bit signless integer |
|  `src`  | 32-bit signless integer |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.ds_swizzle`(ROCDL::DsSwizzleOp)

语法：

```
operation ::= `rocdl.ds_swizzle` $src `,` $offset  attr-dict `:` `(` type($src) `,` type($offset) `)` `->` type($res)
```

#### 操作数：

| Operand  | Description             |
| :------: | ----------------------- |
|  `src`   | 32-bit signless integer |
| `offset` | 32-bit signless integer |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.fmed3`(ROCDL::FMed3Op)

*计算三个浮点/半精度值的中位数*

语法：

```
operation ::= `rocdl.fmed3` $src0 `,` $src1 `,` $src2 attr-dict `:` type($res)
```

使用AMDGPU fmed3内置函数计算三个浮点值的中位数。该操作等效于`max(min(a, b), min(max(a, b), c))`，但采用硬件加速的V_MED3_F16/V_MED3_F32指令以提升性能。

本操作支持标量和向量浮点类型（f16, f32）。

示例：

```mlir
// 标量 f32 中位数
%result = rocdl.fmed3 %a, %b, %c : f32

// 向量 f16 中位数
%result = rocdl.fmed3 %va, %vb, %vc : vector<4xf16>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `src0`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
| `src1`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
| `src2`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

### `rocdl.global.load.async.to.lds.b128`(ROCDL::GlobalLoadAsyncToLDSB128Op)

语法：

```
operation ::= `rocdl.global.load.async.to.lds.b128` $globalPtr `,`  $ldsPtr `,` $offset `,` $aux
              attr-dict `:` qualified(type($globalPtr)) `,` qualified(type($ldsPtr))
```

异步地将128位数据从全局内存指针加载到本地数据共享（LDS）指针。

支持于gfx1250及以上版本。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                          |
| ---------------- | ------------------- | ------------------------------------ |
| `offset`         | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `aux`            | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array |

#### 操作数：

|   Operand   | Description                     |
| :---------: | ------------------------------- |
| `globalPtr` | LLVM pointer in address space 1 |
|  `ldsPtr`   | LLVM pointer in address space 3 |

### `rocdl.global.load.async.to.lds.b32`(ROCDL::GlobalLoadAsyncToLDSB32Op)

语法：

```
operation ::= `rocdl.global.load.async.to.lds.b32` $globalPtr `,`  $ldsPtr `,` $offset `,` $aux
              attr-dict `:` qualified(type($globalPtr)) `,` qualified(type($ldsPtr))
```

从全局内存指针异步加载32位数据至本地数据共享(LDS)指针。

支持gfx1250及以上架构。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                          |
| ---------------- | ------------------- | ------------------------------------ |
| `offset`         | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `aux`            | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array |

#### 操作数：

|   Operand   | Description                     |
| :---------: | ------------------------------- |
| `globalPtr` | LLVM pointer in address space 1 |
|  `ldsPtr`   | LLVM pointer in address space 3 |

### `rocdl.global.load.async.to.lds.b64`(ROCDL::GlobalLoadAsyncToLDSB64Op)

语法：

```
operation ::= `rocdl.global.load.async.to.lds.b64` $globalPtr `,`  $ldsPtr `,` $offset `,` $aux
              attr-dict `:` qualified(type($globalPtr)) `,` qualified(type($ldsPtr))
```

从全局内存指针异步加载64位数据至本地数据共享(LDS)指针。

支持gfx1250及以上架构。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                          |
| ---------------- | ------------------- | ------------------------------------ |
| `offset`         | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `aux`            | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array |

#### 操作数：

|   Operand   | Description                     |
| :---------: | ------------------------------- |
| `globalPtr` | LLVM pointer in address space 1 |
|  `ldsPtr`   | LLVM pointer in address space 3 |

### `rocdl.global.load.async.to.lds.b8`(ROCDL::GlobalLoadAsyncToLDSB8Op)

语法：

```
operation ::= `rocdl.global.load.async.to.lds.b8` $globalPtr `,`  $ldsPtr `,` $offset `,` $aux
              attr-dict `:` qualified(type($globalPtr)) `,` qualified(type($ldsPtr))
```

从全局内存指针异步加载8位数据至本地数据共享(LDS)指针。

支持于gfx1250及以上版本。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                          |
| ---------------- | ------------------- | ------------------------------------ |
| `offset`         | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `aux`            | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array |

#### 操作数：

|   Operand   | Description                     |
| :---------: | ------------------------------- |
| `globalPtr` | LLVM pointer in address space 1 |
|  `ldsPtr`   | LLVM pointer in address space 3 |

### `rocdl.global.load.lds`(ROCDL::GlobalLoadLDSOp)

语法：

```
operation ::= `rocdl.global.load.lds` $globalPtr `,`  $ldsPtr `,` $size `,` $offset `,` $aux
              attr-dict
```

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                          |
| ---------------- | ------------------- | ------------------------------------ |
| `size`           | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `offset`         | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `aux`            | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array |

#### 操作数：

|   Operand   | Description                     |
| :---------: | ------------------------------- |
| `globalPtr` | LLVM pointer in address space 1 |
|  `ldsPtr`   | LLVM pointer in address space 3 |

### `rocdl.global.load.tr.b128`(ROCDL::GlobalLoadTr8_B128)

*从全局内存加载并转置矩阵至寄存器（gfx1250及以上型号）。*

语法：

```
operation ::= `rocdl.global.load.tr.b128` $ptr attr-dict `:` qualified(type($ptr)) `->` type($res)
```

从全局内存加载16位数据矩阵，在行优先与列优先顺序间转置数据，并将结果存储至128位向量寄存器。

在 gfx1250+ 版本可用。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
|  `ptr`  | LLVM pointer in address space 1 |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.global.load.tr.b64`(ROCDL::GlobalLoadTr8_B64)

*从全局内存加载矩阵并转置至寄存器（gfx1250+可用）。*

语法：

```
operation ::= `rocdl.global.load.tr.b64` $ptr attr-dict `:` qualified(type($ptr)) `->` type($res)
```

从全局内存加载8位数据矩阵，在行优先与列优先顺序间转置数据，并将结果存储至64位向量寄存器。

仅支持 gfx1250及以上型号。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
|  `ptr`  | LLVM pointer in address space 1 |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.global.load.tr4.b64`(ROCDL::GlobalLoadTr4_B64)

*从全局内存加载并转置矩阵至寄存器（仅限 gfx1250+ 支持）。*

语法：

```
operation ::= `rocdl.global.load.tr4.b64` $ptr attr-dict `:` qualified(type($ptr)) `->` type($res)
```

从全局内存加载4位数据矩阵，在行优先与列优先顺序间转置数据，并将结果存储至64位向量寄存器。

支持gfx1250+版本。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
|  `ptr`  | LLVM pointer in address space 1 |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.global.load.tr6.b96`(ROCDL::GlobalLoadTr6_B96)

*从全局内存加载矩阵并转置至寄存器（仅支持gfx1250+版本）。*

语法：

```
operation ::= `rocdl.global.load.tr6.b96` $ptr attr-dict `:` qualified(type($ptr)) `->` type($res)
```

从全局内存加载6位数据矩阵，在行优先与列优先顺序间转置数据，并将结果存储至96位向量寄存器。

仅支持 gfx1250 及以上型号。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
|  `ptr`  | LLVM pointer in address space 1 |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.grid.dim.x`(ROCDL::GridDimXOp)

语法：

```
operation ::= `rocdl.grid.dim.x` (`range` $range^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                                                  |
| --------- | ------------------------------- | ------------------------------------------------------------ |
| `range`   | ::mlir::LLVM::ConstantRangeAttr | A range of two integers, corresponding to LLVM's ConstantRange`A pair of two integers, mapping to the ConstantRange structure in LLVM IR, which is allowed to wrap or be empty. The range represented is [Lower, Upper), and is either signed or unsigned depending on context. lower and upper must have the same width. Syntax: `<` `i`(width($lower)) $lower `,` $upper `>` ` |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.grid.dim.y`(ROCDL::GridDimYOp)

语法：

```
operation ::= `rocdl.grid.dim.y` (`range` $range^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                                                  |
| --------- | ------------------------------- | ------------------------------------------------------------ |
| `range`   | ::mlir::LLVM::ConstantRangeAttr | A range of two integers, corresponding to LLVM's ConstantRange`A pair of two integers, mapping to the ConstantRange structure in LLVM IR, which is allowed to wrap or be empty. The range represented is [Lower, Upper), and is either signed or unsigned depending on context. lower and upper must have the same width. Syntax: `<` `i`(width($lower)) $lower `,` $upper `>` ` |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.grid.dim.z`(ROCDL::GridDimZOp)

语法：

```
operation ::= `rocdl.grid.dim.z` (`range` $range^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                                                  |
| --------- | ------------------------------- | ------------------------------------------------------------ |
| `range`   | ::mlir::LLVM::ConstantRangeAttr | A range of two integers, corresponding to LLVM's ConstantRange`A pair of two integers, mapping to the ConstantRange structure in LLVM IR, which is allowed to wrap or be empty. The range represented is [Lower, Upper), and is either signed or unsigned depending on context. lower and upper must have the same width. Syntax: `<` `i`(width($lower)) $lower `,` $upper `>` ` |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.iglp.opt`(ROCDL::IglpOpt)

语法：

```
operation ::= `rocdl.iglp.opt` $variant attr-dict
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `variant` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

### `rocdl.load.to.lds`(ROCDL::LoadToLDSOp)

语法：

```
operation ::= `rocdl.load.to.lds` $globalPtr `,`  $ldsPtr `,` $size `,` $offset `,` $aux
              attr-dict `:` type($globalPtr)
```

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                          |
| ---------------- | ------------------- | ------------------------------------ |
| `size`           | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `offset`         | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `aux`            | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array |

#### 操作数：

|   Operand   | Description                     |
| :---------: | ------------------------------- |
| `globalPtr` | LLVM pointer type               |
|  `ldsPtr`   | LLVM pointer in address space 3 |

### `rocdl.make.buffer.rsrc`(ROCDL::MakeBufferRsrcOp)

语法：

```
operation ::= `rocdl.make.buffer.rsrc` operands attr-dict `:` type($base) `to` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|   Operand    | Description             |
| :----------: | ----------------------- |
|    `base`    | LLVM pointer type       |
|   `stride`   | 16-bit signless integer |
| `numRecords` | 64-bit signless integer |
|   `flags`    | 32-bit signless integer |

#### 结果：

| Result | Description       |
| :----: | ----------------- |
| `res`  | LLVM pointer type |

### `rocdl.mbcnt.hi`(ROCDL::MbcntHiOp)

语法：

```
operation ::= `rocdl.mbcnt.hi` $in0 `,` $in1  attr-dict `:` `(` type($in0) `,` type($in1) `)` `->` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ArgAndResultAttrsOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute   | MLIR Type         | Description                    |
| ----------- | ----------------- | ------------------------------ |
| `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `in0`  | 32-bit signless integer |
|  `in1`  | 32-bit signless integer |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.mbcnt.lo`(ROCDL::MbcntLoOp)

语法：

```
operation ::= `rocdl.mbcnt.lo` $in0 `,` $in1  attr-dict `:` `(` type($in0) `,` type($in1) `)` `->` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ArgAndResultAttrsOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute   | MLIR Type         | Description                    |
| ----------- | ----------------- | ------------------------------ |
| `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `in0`  | 32-bit signless integer |
|  `in1`  | 32-bit signless integer |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.mfma.f32.16x16x16bf16.1k`(ROCDL::mfma_f32_16x16x16bf16_1k)

语法：

```
operation ::= `rocdl.mfma.f32.16x16x16bf16.1k` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.16x16x16f16`(ROCDL::mfma_f32_16x16x16f16)

语法：

```
operation ::= `rocdl.mfma.f32.16x16x16f16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.16x16x1f32`(ROCDL::mfma_f32_16x16x1f32)

语法：

```
operation ::= `rocdl.mfma.f32.16x16x1f32` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.16x16x2bf16`(ROCDL::mfma_f32_16x16x2bf16)

语法：

```
operation ::= `rocdl.mfma.f32.16x16x2bf16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.16x16x32.bf16`(ROCDL::mfma_f32_16x16x32_bf16)

语法：

```
operation ::= `rocdl.mfma.f32.16x16x32.bf16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.16x16x32.bf8.bf8`(ROCDL::mfma_f32_16x16x32_bf8_bf8)

语法：

```
operation ::= `rocdl.mfma.f32.16x16x32.bf8.bf8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.16x16x32.bf8.fp8`(ROCDL::mfma_f32_16x16x32_bf8_fp8)

语法：

```
operation ::= `rocdl.mfma.f32.16x16x32.bf8.fp8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.16x16x32.f16`(ROCDL::mfma_f32_16x16x32_f16)

语法：

```
operation ::= `rocdl.mfma.f32.16x16x32.f16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.16x16x32.fp8.bf8`(ROCDL::mfma_f32_16x16x32_fp8_bf8)

语法：

```
operation ::= `rocdl.mfma.f32.16x16x32.fp8.bf8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.16x16x32.fp8.fp8`(ROCDL::mfma_f32_16x16x32_fp8_fp8)

语法：

```
operation ::= `rocdl.mfma.f32.16x16x32.fp8.fp8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.16x16x4bf16.1k`(ROCDL::mfma_f32_16x16x4bf16_1k)

语法：

```
operation ::= `rocdl.mfma.f32.16x16x4bf16.1k` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.16x16x4f16`(ROCDL::mfma_f32_16x16x4f16)

语法：

```
operation ::= `rocdl.mfma.f32.16x16x4f16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.16x16x4f32`(ROCDL::mfma_f32_16x16x4f32)

语法：

```
operation ::= `rocdl.mfma.f32.16x16x4f32` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.16x16x8.xf32`(ROCDL::mfma_f32_16x16x8_xf32)

语法：

```
operation ::= `rocdl.mfma.f32.16x16x8.xf32` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.16x16x8bf16`(ROCDL::mfma_f32_16x16x8bf16)

语法：

```
operation ::= `rocdl.mfma.f32.16x16x8bf16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.32x32x16.bf16`(ROCDL::mfma_f32_32x32x16_bf16)

语法：

```
operation ::= `rocdl.mfma.f32.32x32x16.bf16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.32x32x16.bf8.bf8`(ROCDL::mfma_f32_32x32x16_bf8_bf8)

语法：

```
operation ::= `rocdl.mfma.f32.32x32x16.bf8.bf8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.32x32x16.bf8.fp8`(ROCDL::mfma_f32_32x32x16_bf8_fp8)

语法：

```
operation ::= `rocdl.mfma.f32.32x32x16.bf8.fp8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.32x32x16.f16`(ROCDL::mfma_f32_32x32x16_f16)

语法：

```
operation ::= `rocdl.mfma.f32.32x32x16.f16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.32x32x16.fp8.bf8`(ROCDL::mfma_f32_32x32x16_fp8_bf8)

语法：

```
operation ::= `rocdl.mfma.f32.32x32x16.fp8.bf8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.32x32x16.fp8.fp8`(ROCDL::mfma_f32_32x32x16_fp8_fp8)

语法：

```
operation ::= `rocdl.mfma.f32.32x32x16.fp8.fp8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.32x32x1f32`(ROCDL::mfma_f32_32x32x1f32)

语法：

```
operation ::= `rocdl.mfma.f32.32x32x1f32` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.32x32x2bf16`(ROCDL::mfma_f32_32x32x2bf16)

语法：

```
operation ::= `rocdl.mfma.f32.32x32x2bf16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.32x32x2f32`(ROCDL::mfma_f32_32x32x2f32)

语法：

```
operation ::= `rocdl.mfma.f32.32x32x2f32` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.32x32x4.xf32`(ROCDL::mfma_f32_32x32x4_xf32)

语法：

```
operation ::= `rocdl.mfma.f32.32x32x4.xf32` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.32x32x4bf16`(ROCDL::mfma_f32_32x32x4bf16)

语法：

```
operation ::= `rocdl.mfma.f32.32x32x4bf16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.32x32x4bf16.1k`(ROCDL::mfma_f32_32x32x4bf16_1k)

语法：

```
operation ::= `rocdl.mfma.f32.32x32x4bf16.1k` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.32x32x4f16`(ROCDL::mfma_f32_32x32x4f16)

语法：

```
operation ::= `rocdl.mfma.f32.32x32x4f16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.32x32x8bf16.1k`(ROCDL::mfma_f32_32x32x8bf16_1k)

语法：

```
operation ::= `rocdl.mfma.f32.32x32x8bf16.1k` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.32x32x8f16`(ROCDL::mfma_f32_32x32x8f16)

语法：

```
operation ::= `rocdl.mfma.f32.32x32x8f16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.4x4x1f32`(ROCDL::mfma_f32_4x4x1f32)

语法：

```
operation ::= `rocdl.mfma.f32.4x4x1f32` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.4x4x2bf16`(ROCDL::mfma_f32_4x4x2bf16)

语法：

```
operation ::= `rocdl.mfma.f32.4x4x2bf16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.4x4x4bf16.1k`(ROCDL::mfma_f32_4x4x4bf16_1k)

语法：

```
operation ::= `rocdl.mfma.f32.4x4x4bf16.1k` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f32.4x4x4f16`(ROCDL::mfma_f32_4x4x4f16)

语法：

```
operation ::= `rocdl.mfma.f32.4x4x4f16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f64.16x16x4f64`(ROCDL::mfma_f64_16x16x4f64)

语法：

```
operation ::= `rocdl.mfma.f64.16x16x4f64` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.f64.4x4x4f64`(ROCDL::mfma_f64_4x4x4f64)

语法：

```
operation ::= `rocdl.mfma.f64.4x4x4f64` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.i32.16x16x16i8`(ROCDL::mfma_i32_16x16x16i8)

语法：

```
operation ::= `rocdl.mfma.i32.16x16x16i8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.i32.16x16x32.i8`(ROCDL::mfma_i32_16x16x32_i8)

语法：

```
operation ::= `rocdl.mfma.i32.16x16x32.i8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.i32.16x16x4i8`(ROCDL::mfma_i32_16x16x4i8)

语法：

```
operation ::= `rocdl.mfma.i32.16x16x4i8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.i32.16x16x64.i8`(ROCDL::mfma_i32_16x16x64_i8)

语法：

```
operation ::= `rocdl.mfma.i32.16x16x64.i8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.i32.32x32x16.i8`(ROCDL::mfma_i32_32x32x16_i8)

语法：

```
operation ::= `rocdl.mfma.i32.32x32x16.i8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.i32.32x32x32.i8`(ROCDL::mfma_i32_32x32x32_i8)

语法：

```
operation ::= `rocdl.mfma.i32.32x32x32.i8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.i32.32x32x4i8`(ROCDL::mfma_i32_32x32x4i8)

语法：

```
operation ::= `rocdl.mfma.i32.32x32x4i8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.i32.32x32x8i8`(ROCDL::mfma_i32_32x32x8i8)

语法：

```
operation ::= `rocdl.mfma.i32.32x32x8i8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.i32.4x4x4i8`(ROCDL::mfma_i32_4x4x4i8)

语法：

```
operation ::= `rocdl.mfma.i32.4x4x4i8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.scale.f32.16x16x128.f8f6f4`(ROCDL::mfma_scale_f32_16x16x128_f8f6f4)

语法：

```
operation ::= `rocdl.mfma.scale.f32.16x16x128.f8f6f4` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.mfma.scale.f32.32x32x64.f8f6f4`(ROCDL::mfma_scale_f32_32x32x64_f8f6f4)

语法：

```
operation ::= `rocdl.mfma.scale.f32.32x32x64.f8f6f4` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.permlane16.swap`(ROCDL::Permlane16SwapOp)

语法：

```
operation ::= `rocdl.permlane16.swap` attr-dict $old `,` $src `,` $fi `,` $boundControl `:` `(` type($old) `,` type($src) `)` `->` type($res)
```

对给定操作数执行`permlane16.swap`操作，将 $fi 指定的置换规则应用于给定输入。

#### 属性：

| Attribute      | MLIR Type           | Description                      |
| -------------- | ------------------- | -------------------------------- |
| `fi`           | ::mlir::IntegerAttr | 1-bit signless integer attribute |
| `boundControl` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `old`  | 32-bit signless integer |
|  `src`  | 32-bit signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | LLVM dialect-compatible struct of 32-bit signless integerand32-bit signless integer |

### `rocdl.permlane32.swap`(ROCDL::Permlane32SwapOp)

语法：

```
operation ::= `rocdl.permlane32.swap` attr-dict $old `,` $src `,` $fi `,` $boundControl `:` `(` type($old) `,` type($src) `)` `->` type($res)
```

对给定操作数执行`permlane32.swap`操作，将 $fi 指定的置换应用于给定输入。

#### 属性：

| Attribute      | MLIR Type           | Description                      |
| -------------- | ------------------- | -------------------------------- |
| `fi`           | ::mlir::IntegerAttr | 1-bit signless integer attribute |
| `boundControl` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `old`  | 32-bit signless integer |
|  `src`  | 32-bit signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | LLVM dialect-compatible struct of 32-bit signless integerand32-bit signless integer |

### `rocdl.permlanex16`(ROCDL::PermlaneX16Op)

语法：

```
operation ::= `rocdl.permlanex16` attr-dict $old `,` $src0 `,` $src1 `,` $src2 `,` $fi `,` $boundControl `:` type($src0) `,` type($src1)
```

对给定操作数执行`permlanex16`操作，对给定输入应用$fi指定的置换。

#### 属性：

| Attribute      | MLIR Type           | Description                      |
| -------------- | ------------------- | -------------------------------- |
| `fi`           | ::mlir::IntegerAttr | 1-bit signless integer attribute |
| `boundControl` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description                  |
| :-----: | ---------------------------- |
|  `old`  | LLVM dialect-compatible type |
| `src0`  | LLVM dialect-compatible type |
| `src1`  | LLVM dialect-compatible type |
| `src2`  | LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.raw.buffer.atomic.cmpswap`(ROCDL::RawBufferAtomicCmpSwap)

语法：

```
operation ::= `rocdl.raw.buffer.atomic.cmpswap` attr-dict `(` operands `)` `:` type($res) `,` type($rsrc)
```

#### 操作数：

|  Operand  | Description                  |
| :-------: | ---------------------------- |
|   `src`   | LLVM dialect-compatible type |
|   `cmp`   | LLVM dialect-compatible type |
|  `rsrc`   | LLVM dialect-compatible type |
| `offset`  | 32-bit signless integer      |
| `soffset` | 32-bit signless integer      |
|   `aux`   | 32-bit signless integer      |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.raw.buffer.atomic.fadd`(ROCDL::RawBufferAtomicFAddOp)

#### 操作数：

|  Operand  | Description                  |
| :-------: | ---------------------------- |
|  `vdata`  | LLVM dialect-compatible type |
|  `rsrc`   | LLVM dialect-compatible type |
| `offset`  | LLVM dialect-compatible type |
| `soffset` | LLVM dialect-compatible type |
|   `aux`   | LLVM dialect-compatible type |

### `rocdl.raw.buffer.atomic.fmax`(ROCDL::RawBufferAtomicFMaxOp)

#### 操作数：

|  Operand  | Description                  |
| :-------: | ---------------------------- |
|  `vdata`  | LLVM dialect-compatible type |
|  `rsrc`   | LLVM dialect-compatible type |
| `offset`  | LLVM dialect-compatible type |
| `soffset` | LLVM dialect-compatible type |
|   `aux`   | LLVM dialect-compatible type |

### `rocdl.raw.buffer.atomic.smax`(ROCDL::RawBufferAtomicSMaxOp)

#### 操作数：

|  Operand  | Description                  |
| :-------: | ---------------------------- |
|  `vdata`  | LLVM dialect-compatible type |
|  `rsrc`   | LLVM dialect-compatible type |
| `offset`  | LLVM dialect-compatible type |
| `soffset` | LLVM dialect-compatible type |
|   `aux`   | LLVM dialect-compatible type |

### `rocdl.raw.buffer.atomic.umin`(ROCDL::RawBufferAtomicUMinOp)

#### 操作数：

|  Operand  | Description                  |
| :-------: | ---------------------------- |
|  `vdata`  | LLVM dialect-compatible type |
|  `rsrc`   | LLVM dialect-compatible type |
| `offset`  | LLVM dialect-compatible type |
| `soffset` | LLVM dialect-compatible type |
|   `aux`   | LLVM dialect-compatible type |

### `rocdl.raw.buffer.load`(ROCDL::RawBufferLoadOp)

#### 操作数：

|  Operand  | Description                  |
| :-------: | ---------------------------- |
|  `rsrc`   | LLVM dialect-compatible type |
| `offset`  | LLVM dialect-compatible type |
| `soffset` | LLVM dialect-compatible type |
|   `aux`   | LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.raw.buffer.store`(ROCDL::RawBufferStoreOp)

#### 操作数：

|  Operand  | Description                  |
| :-------: | ---------------------------- |
|  `vdata`  | LLVM dialect-compatible type |
|  `rsrc`   | LLVM dialect-compatible type |
| `offset`  | LLVM dialect-compatible type |
| `soffset` | LLVM dialect-compatible type |
|   `aux`   | LLVM dialect-compatible type |

### `rocdl.raw.ptr.buffer.atomic.cmpswap`(ROCDL::RawPtrBufferAtomicCmpSwap)

语法：

```
operation ::= `rocdl.raw.ptr.buffer.atomic.cmpswap` operands attr-dict `:` type($res)
```

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

|  Operand  | Description                     |
| :-------: | ------------------------------- |
|   `src`   | LLVM dialect-compatible type    |
|   `cmp`   | LLVM dialect-compatible type    |
|  `rsrc`   | LLVM pointer in address space 8 |
| `offset`  | 32-bit signless integer         |
| `soffset` | 32-bit signless integer         |
|   `aux`   | 32-bit signless integer         |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.raw.ptr.buffer.atomic.fadd`(ROCDL::RawPtrBufferAtomicFaddOp)

语法：

```
operation ::= `rocdl.raw.ptr.buffer.atomic.fadd` operands attr-dict `:` type($vdata)
```

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

|  Operand  | Description                     |
| :-------: | ------------------------------- |
|  `vdata`  | LLVM dialect-compatible type    |
|  `rsrc`   | LLVM pointer in address space 8 |
| `offset`  | 32-bit signless integer         |
| `soffset` | 32-bit signless integer         |
|   `aux`   | 32-bit signless integer         |

### `rocdl.raw.ptr.buffer.atomic.fmax`(ROCDL::RawPtrBufferAtomicFmaxOp)

语法：

```
operation ::= `rocdl.raw.ptr.buffer.atomic.fmax` operands attr-dict `:` type($vdata)
```

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

|  Operand  | Description                     |
| :-------: | ------------------------------- |
|  `vdata`  | LLVM dialect-compatible type    |
|  `rsrc`   | LLVM pointer in address space 8 |
| `offset`  | 32-bit signless integer         |
| `soffset` | 32-bit signless integer         |
|   `aux`   | 32-bit signless integer         |

### `rocdl.raw.ptr.buffer.atomic.smax`(ROCDL::RawPtrBufferAtomicSmaxOp)

语法：

```
operation ::= `rocdl.raw.ptr.buffer.atomic.smax` operands attr-dict `:` type($vdata)
```

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

|  Operand  | Description                     |
| :-------: | ------------------------------- |
|  `vdata`  | LLVM dialect-compatible type    |
|  `rsrc`   | LLVM pointer in address space 8 |
| `offset`  | 32-bit signless integer         |
| `soffset` | 32-bit signless integer         |
|   `aux`   | 32-bit signless integer         |

### `rocdl.raw.ptr.buffer.atomic.umin`(ROCDL::RawPtrBufferAtomicUminOp)

语法：

```
operation ::= `rocdl.raw.ptr.buffer.atomic.umin` operands attr-dict `:` type($vdata)
```

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

|  Operand  | Description                     |
| :-------: | ------------------------------- |
|  `vdata`  | LLVM dialect-compatible type    |
|  `rsrc`   | LLVM pointer in address space 8 |
| `offset`  | 32-bit signless integer         |
| `soffset` | 32-bit signless integer         |
|   `aux`   | 32-bit signless integer         |

### `rocdl.raw.ptr.buffer.load`(ROCDL::RawPtrBufferLoadOp)

语法：

```
operation ::= `rocdl.raw.ptr.buffer.load` operands attr-dict `:` type($res)
```

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

|  Operand  | Description                     |
| :-------: | ------------------------------- |
|  `rsrc`   | LLVM pointer in address space 8 |
| `offset`  | 32-bit signless integer         |
| `soffset` | 32-bit signless integer         |
|   `aux`   | 32-bit signless integer         |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.raw.ptr.buffer.load.lds`(ROCDL::RawPtrBufferLoadLdsOp)

语法：

```
operation ::= `rocdl.raw.ptr.buffer.load.lds` operands attr-dict
```

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

|  Operand  | Description                     |
| :-------: | ------------------------------- |
|  `rsrc`   | LLVM pointer in address space 8 |
| `ldsPtr`  | LLVM pointer in address space 3 |
|  `size`   | 32-bit signless integer         |
| `voffset` | 32-bit signless integer         |
| `soffset` | 32-bit signless integer         |
| `offset`  | 32-bit signless integer         |
|   `aux`   | 32-bit signless integer         |

### `rocdl.raw.ptr.buffer.store`(ROCDL::RawPtrBufferStoreOp)

语法：

```
operation ::= `rocdl.raw.ptr.buffer.store` operands attr-dict `:` type($vdata)
```

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type         | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `alias_scopes`   | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr | LLVM dialect TBAA tag metadata array |

#### 操作数：

|  Operand  | Description                     |
| :-------: | ------------------------------- |
|  `vdata`  | LLVM dialect-compatible type    |
|  `rsrc`   | LLVM pointer in address space 8 |
| `offset`  | 32-bit signless integer         |
| `soffset` | 32-bit signless integer         |
|   `aux`   | 32-bit signless integer         |

### `rocdl.readfirstlane`(ROCDL::ReadfirstlaneOp)

*获取第一个活跃lane中的值。*

语法：

```
operation ::= `rocdl.readfirstlane` $src attr-dict `:` type($res)
```

返回输入操作数最低活跃lane中的值。

#### 操作数：

| Operand | Description                  |
| :-----: | ---------------------------- |
|  `src`  | LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.readlane`(ROCDL::ReadlaneOp)

*获取特定lane中的值。*

语法：

```
operation ::= `rocdl.readlane` $src0 `,` $src1  attr-dict `:` `(` type($src0) `,` type($src1) `)` `->` type($res)
```

从输入`src0`中获取lane`src1`中的值。

#### 操作数：

| Operand | Description                  |
| :-----: | ---------------------------- |
| `src0`  | LLVM dialect-compatible type |
| `src1`  | 32-bit signless integer      |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.s.barrier`(ROCDL::SBarrierOp)

语法：

```
operation ::= `rocdl.s.barrier` attr-dict
```

### `rocdl.s.barrier.init`(ROCDL::BarrierInitOp)

语法：

```
operation ::= `rocdl.s.barrier.init` $ptr `,` $id attr-dict
```

仅支持 gfx1250+ 架构。

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `id`      | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
|  `ptr`  | LLVM pointer in address space 3 |

### `rocdl.s.barrier.join`(ROCDL::BarrierJoinOp)

语法：

```
operation ::= `rocdl.s.barrier.join` $ptr attr-dict
```

支持gfx1250及以上版本。

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
|  `ptr`  | LLVM pointer in address space 3 |

### `rocdl.s.barrier.leave`(ROCDL::BarrierLeaveOp)

语法：

```
operation ::= `rocdl.s.barrier.leave` $id attr-dict
```

支持gfx1250及以上版本。

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `id`      | ::mlir::IntegerAttr | 16-bit signless integer attribute |

### `rocdl.s.barrier.signal`(ROCDL::BarrierSignalOp)

语法：

```
operation ::= `rocdl.s.barrier.signal` $id attr-dict
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `id`      | ::mlir::IntegerAttr | 32-bit signless integer attribute |

### `rocdl.s.barrier.signal.isfirst`(ROCDL::BarrierSignalIsfirstOp)

语法：

```
operation ::= `rocdl.s.barrier.signal.isfirst` $id attr-dict `:` type($res)
```

适用于 gfx1250 及以上版本。

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `id`      | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 结果：

| Result | Description            |
| :----: | ---------------------- |
| `res`  | 1-bit signless integer |

### `rocdl.s.barrier.signal.var`(ROCDL::BarrierSignalVarOp)

语法：

```
operation ::= `rocdl.s.barrier.signal.var` $ptr `,` $id attr-dict
```

支持 gfx1250 及以上版本。

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `id`      | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
|  `ptr`  | LLVM pointer in address space 3 |

### `rocdl.s.barrier.wait`(ROCDL::BarrierWaitOp)

语法：

```
operation ::= `rocdl.s.barrier.wait` $id attr-dict
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `id`      | ::mlir::IntegerAttr | 16-bit signless integer attribute |

### `rocdl.s.get.barrier.state`(ROCDL::GetBarrierStateOp)

语法：

```
operation ::= `rocdl.s.get.barrier.state` $id attr-dict `:` type($res)
```

仅适用于gfx1250及以上版本。

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `id`      | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.s.get.named.barrier.state`(ROCDL::GetNamedBarrierStateOp)

语法：

```
operation ::= `rocdl.s.get.named.barrier.state` $ptr attr-dict `:` type($res)
```

支持 gfx1250 及以上版本。

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
|  `ptr`  | LLVM pointer in address space 3 |

#### 结果：

| Result | Description             |
| :----: | ----------------------- |
| `res`  | 32-bit signless integer |

### `rocdl.s.setprio`(ROCDL::SetPrioOp)

语法：

```
operation ::= `rocdl.s.setprio` $priority attr-dict
```

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `priority` | ::mlir::IntegerAttr | 16-bit signless integer attribute |

### `rocdl.s.sleep`(ROCDL::SSleepOp)

语法：

```
operation ::= `rocdl.s.sleep` attr-dict $count
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `count`   | ::mlir::IntegerAttr | 32-bit signless integer attribute |

### `rocdl.s.wait.asynccnt`(ROCDL::WaitAsynccntOp)

*等待直到 ASYNCCNT 小于或等于`count`*

语法：

```
operation ::= `rocdl.s.wait.asynccnt` $count attr-dict
```

等待指定的计数器小于或等于 `count` 后再继续。

支持于 gfx1250 及以上版本

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `count`   | ::mlir::IntegerAttr | 16-bit signless integer attribute |

### `rocdl.s.wait.dscnt`(ROCDL::WaitDscntOp)

*等待直到 DSCNT 小于或等于`count`*

语法：

```
operation ::= `rocdl.s.wait.dscnt` $count attr-dict
```

等待指定计数器小于或等于`count`后继续执行。

支持gfx12+版本。

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `count`   | ::mlir::IntegerAttr | 16-bit signless integer attribute |

### `rocdl.s.wait.expcnt`(ROCDL::WaitExpcntOp)

*等待直到EXPCNT小于或等于`count`*

语法：

```
operation ::= `rocdl.s.wait.expcnt` $count attr-dict
```

等待指定计数器小于或等于`count`后继续执行。

仅支持 gfx12 及以上版本。

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `count`   | ::mlir::IntegerAttr | 16-bit signless integer attribute |

### `rocdl.s.wait.loadcnt`(ROCDL::WaitLoadcntOp)

*等待直到 LOADCNT 小于或等于`count`*

语法：

```
operation ::= `rocdl.s.wait.loadcnt` $count attr-dict
```

等待指定计数器小于或等于`count`后继续执行。

支持gfx12及以上版本。

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `count`   | ::mlir::IntegerAttr | 16-bit signless integer attribute |

### `rocdl.s.wait.storecnt`(ROCDL::WaitStorecntOp)

*等待直到 STORECNT 小于或等于`count`*

语法：

```
operation ::= `rocdl.s.wait.storecnt` $count attr-dict
```

等待指定计数器小于或等于`count`后继续执行。

支持于 gfx12+ 版本。

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `count`   | ::mlir::IntegerAttr | 16-bit signless integer attribute |

### `rocdl.s.wait.tensorcnt`(ROCDL::WaitTensorcntOp)

*等待直到 TENSORCNT 小于或等于`count`*

语法：

```
operation ::= `rocdl.s.wait.tensorcnt` $count attr-dict
```

等待指定计数器小于或等于`count`后继续执行。

支持gfx1250及以上版本。

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `count`   | ::mlir::IntegerAttr | 16-bit signless integer attribute |

### `rocdl.s.waitcnt`(ROCDL::SWaitcntOp)

语法：

```
operation ::= `rocdl.s.waitcnt` attr-dict $bitfield
```

#### 属性：

| Attribute  | MLIR Type           | Description                       |
| ---------- | ------------------- | --------------------------------- |
| `bitfield` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

### `rocdl.sched.barrier`(ROCDL::SchedBarrier)

语法：

```
operation ::= `rocdl.sched.barrier` $mask attr-dict
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `mask`    | ::mlir::IntegerAttr | 32-bit signless integer attribute |

### `rocdl.sched.group.barrier`(ROCDL::SchedGroupBarrier)

语法：

```
operation ::= `rocdl.sched.group.barrier` $mask `,` $size `,` $groupId attr-dict
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `mask`    | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `size`    | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `groupId` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

### `rocdl.smfmac.f32.16x16x32.bf16`(ROCDL::smfmac_f32_16x16x32_bf16)

语法：

```
operation ::= `rocdl.smfmac.f32.16x16x32.bf16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.smfmac.f32.16x16x32.f16`(ROCDL::smfmac_f32_16x16x32_f16)

语法：

```
operation ::= `rocdl.smfmac.f32.16x16x32.f16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.smfmac.f32.16x16x64.bf8.bf8`(ROCDL::smfmac_f32_16x16x64_bf8_bf8)

语法：

```
operation ::= `rocdl.smfmac.f32.16x16x64.bf8.bf8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.smfmac.f32.16x16x64.bf8.fp8`(ROCDL::smfmac_f32_16x16x64_bf8_fp8)

语法：

```
operation ::= `rocdl.smfmac.f32.16x16x64.bf8.fp8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.smfmac.f32.16x16x64.fp8.bf8`(ROCDL::smfmac_f32_16x16x64_fp8_bf8)

语法：

```
operation ::= `rocdl.smfmac.f32.16x16x64.fp8.bf8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.smfmac.f32.16x16x64.fp8.fp8`(ROCDL::smfmac_f32_16x16x64_fp8_fp8)

语法：

```
operation ::= `rocdl.smfmac.f32.16x16x64.fp8.fp8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.smfmac.f32.32x32x16.bf16`(ROCDL::smfmac_f32_32x32x16_bf16)

语法：

```
operation ::= `rocdl.smfmac.f32.32x32x16.bf16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.smfmac.f32.32x32x16.f16`(ROCDL::smfmac_f32_32x32x16_f16)

语法：

```
operation ::= `rocdl.smfmac.f32.32x32x16.f16` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.smfmac.f32.32x32x32.bf8.bf8`(ROCDL::smfmac_f32_32x32x32_bf8_bf8)

语法：

```
operation ::= `rocdl.smfmac.f32.32x32x32.bf8.bf8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.smfmac.f32.32x32x32.bf8.fp8`(ROCDL::smfmac_f32_32x32x32_bf8_fp8)

语法：

```
operation ::= `rocdl.smfmac.f32.32x32x32.bf8.fp8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.smfmac.f32.32x32x32.fp8.bf8`(ROCDL::smfmac_f32_32x32x32_fp8_bf8)

语法：

```
operation ::= `rocdl.smfmac.f32.32x32x32.fp8.bf8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.smfmac.f32.32x32x32.fp8.fp8`(ROCDL::smfmac_f32_32x32x32_fp8_fp8)

语法：

```
operation ::= `rocdl.smfmac.f32.32x32x32.fp8.fp8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.smfmac.i32.16x16x64.i8`(ROCDL::smfmac_i32_16x16x64_i8)

语法：

```
operation ::= `rocdl.smfmac.i32.16x16x64.i8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.smfmac.i32.32x32x32.i8`(ROCDL::smfmac_i32_32x32x32_i8)

语法：

```
operation ::= `rocdl.smfmac.i32.32x32x32.i8` $args attr-dict `:` functional-type($args, $res)
```

#### 操作数：

| Operand | Description                              |
| :-----: | ---------------------------------------- |
| `args`  | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.tensor.load.to.lds`(ROCDL::TensorLoadToLDSOp)

*ROCDL 张量与 LDS 之间加载/存储的基类。*

语法：

```
operation ::= `rocdl.tensor.load.to.lds` attr-dict operands `cachepolicy` $cachePolicy `:` type($dgroup0) `,` type($dgroup1)
```

在全局内存与LDS间移动张量数据块。数据块由 dgroupdescriptors.4 dgroup 描述符描述，允许移动最多 5 维张量。$cachePolicy描述内存作用域并指示预期数据复用程度。

本操作适用于gfx1250+架构。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                          |
| ---------------- | ------------------- | ------------------------------------ |
| `cachePolicy`    | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array |

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `dgroup0` | fixed-length vector of 32-bit signless integer values of length 4 |
| `dgroup1` | fixed-length vector of 32-bit signless integer values of length 8 |
| `dgroup2` | fixed-length vector of 32-bit signless integer values of length 4 |
| `dgroup3` | fixed-length vector of 32-bit signless integer values of length 4 |

### `rocdl.tensor.load.to.lds.d2`(ROCDL::TensorLoadToLDSD2Op)

*ROCDL张量与LDS（D2变体）间加载/存储的基类。*

语法：

```
operation ::= `rocdl.tensor.load.to.lds.d2` attr-dict operands `cachepolicy` $cachePolicy `:` type($dgroup0) `,` type($dgroup1)
```

在全局内存与LDS之间移动张量数据块。数据块由 dgroupdescriptors.2 dgroup 描述符描述，允许移动最多 2D 张量。$cachePolicy描述内存作用域并指示预期数据复用程度。

此操作适用于gfx1250+架构。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                          |
| ---------------- | ------------------- | ------------------------------------ |
| `cachePolicy`    | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array |

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `dgroup0` | fixed-length vector of 32-bit signless integer values of length 4 |
| `dgroup1` | fixed-length vector of 32-bit signless integer values of length 8 |

### `rocdl.tensor.store.from.lds`(ROCDL::TensorStoreFromLDSOp)

*ROCDL 张量与 LDS 间加载/存储的基类。*

语法：

```
operation ::= `rocdl.tensor.store.from.lds` attr-dict operands `cachepolicy` $cachePolicy `:` type($dgroup0) `,` type($dgroup1)
```

在全局内存与LDS间移动张量数据块。数据块由 dgroupdescriptors.4 dgroup 描述符描述，允许移动最多 5 维张量。$cachePolicy描述内存作用域并指示预期数据复用程度。

此操作适用于 gfx1250+ 架构。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                          |
| ---------------- | ------------------- | ------------------------------------ |
| `cachePolicy`    | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array |

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `dgroup0` | fixed-length vector of 32-bit signless integer values of length 4 |
| `dgroup1` | fixed-length vector of 32-bit signless integer values of length 8 |
| `dgroup2` | fixed-length vector of 32-bit signless integer values of length 4 |
| `dgroup3` | fixed-length vector of 32-bit signless integer values of length 4 |

### `rocdl.tensor.store.from.lds.d2`(ROCDL::TensorStoreFromLDSD2Op)

*ROCDL张量从/向LDS（D2变体）加载/存储的基类。*

语法：

```
operation ::= `rocdl.tensor.store.from.lds.d2` attr-dict operands `cachepolicy` $cachePolicy `:` type($dgroup0) `,` type($dgroup1)
```

在全局内存与LDS间移动张量数据块。数据块由 dgroupdescriptors.2 dgroup 描述符描述，允许移动最多 2D 张量。$cachePolicy描述内存作用域及预期数据复用程度。

本操作适用于gfx1250+架构。

Interfaces: `AliasAnalysisOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                          |
| ---------------- | ------------------- | ------------------------------------ |
| `cachePolicy`    | ::mlir::IntegerAttr | 32-bit signless integer attribute    |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array       |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array |

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `dgroup0` | fixed-length vector of 32-bit signless integer values of length 4 |
| `dgroup1` | fixed-length vector of 32-bit signless integer values of length 8 |

### `rocdl.update.dpp`(ROCDL::DPPUpdateOp)

语法：

```
operation ::= `rocdl.update.dpp` attr-dict $old `,` $src `with` $dppCtrl `,` $rowMask `,` $bankMask `,` $boundCtrl `:` type($src)
```

#### 属性：

| Attribute   | MLIR Type           | Description                       |
| ----------- | ------------------- | --------------------------------- |
| `dppCtrl`   | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `rowMask`   | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `bankMask`  | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `boundCtrl` | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                  |
| :-----: | ---------------------------- |
|  `old`  | LLVM dialect-compatible type |
|  `src`  | LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.wavefrontsize`(ROCDL::WavefrontSizeOp)

语法：

```
operation ::= `rocdl.wavefrontsize` (`range` $range^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                                                  |
| --------- | ------------------------------- | ------------------------------------------------------------ |
| `range`   | ::mlir::LLVM::ConstantRangeAttr | A range of two integers, corresponding to LLVM's ConstantRange`A pair of two integers, mapping to the ConstantRange structure in LLVM IR, which is allowed to wrap or be empty. The range represented is [Lower, Upper), and is either signed or unsigned depending on context. lower and upper must have the same width. Syntax: `<` `i`(width($lower)) $lower `,` $upper `>` ` |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.wmma.bf16.16x16x16.bf16`(ROCDL::wmma_bf16_16x16x16_bf16)

语法：

```
operation ::= `rocdl.wmma.bf16.16x16x16.bf16` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                      |
| --------- | ------------------- | -------------------------------- |
| `opsel`   | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description                                          |
| :-----: | ---------------------------------------------------- |
|   `a`   | integer or LLVM dialect-compatible vector of integer |
|   `b`   | integer or LLVM dialect-compatible vector of integer |
|   `c`   | integer or LLVM dialect-compatible vector of integer |

#### 结果：

| Result | Description                                          |
| :----: | ---------------------------------------------------- |
| `res`  | integer or LLVM dialect-compatible vector of integer |

### `rocdl.wmma.bf16.16x16x32.bf16`(ROCDL::wmma_bf16_16x16x32_bf16)

语法：

```
operation ::= `rocdl.wmma.bf16.16x16x32.bf16` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `signA`   | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `signB`   | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | bfloat16 type or LLVM dialect-compatible vector of bfloat16 type |
|   `b`   | bfloat16 type or LLVM dialect-compatible vector of bfloat16 type |
|   `c`   | bfloat16 type or LLVM dialect-compatible vector of bfloat16 type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | bfloat16 type or LLVM dialect-compatible vector of bfloat16 type |

### `rocdl.wmma.bf16f32.16x16x32.bf16`(ROCDL::wmma_bf16f32_16x16x32_bf16)

语法：

```
operation ::= `rocdl.wmma.bf16f32.16x16x32.bf16` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `signA`   | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `signB`   | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | bfloat16 type or LLVM dialect-compatible vector of bfloat16 type |
|   `b`   | bfloat16 type or LLVM dialect-compatible vector of bfloat16 type |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | bfloat16 type or LLVM dialect-compatible vector of bfloat16 type |

### `rocdl.wmma.f16.16x16x128.bf8_bf8`(ROCDL::wmma_f16_16x16x128_bf8_bf8)

语法：

```
operation ::= `rocdl.wmma.f16.16x16x128.bf8_bf8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

### `rocdl.wmma.f16.16x16x128.bf8_fp8`(ROCDL::wmma_f16_16x16x128_bf8_fp8)

语法：

```
operation ::= `rocdl.wmma.f16.16x16x128.bf8_fp8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

### `rocdl.wmma.f16.16x16x128.fp8_bf8`(ROCDL::wmma_f16_16x16x128_fp8_bf8)

语法：

```
operation ::= `rocdl.wmma.f16.16x16x128.fp8_bf8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

### `rocdl.wmma.f16.16x16x128.fp8_fp8`(ROCDL::wmma_f16_16x16x128_fp8_fp8)

语法：

```
operation ::= `rocdl.wmma.f16.16x16x128.fp8_fp8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

### `rocdl.wmma.f16.16x16x16.f16`(ROCDL::wmma_f16_16x16x16_f16)

语法：

```
operation ::= `rocdl.wmma.f16.16x16x16.f16` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                      |
| --------- | ------------------- | -------------------------------- |
| `opsel`   | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |
|   `b`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |
|   `c`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

### `rocdl.wmma.f16.16x16x32.f16`(ROCDL::wmma_f16_16x16x32_f16)

语法：

```
operation ::= `rocdl.wmma.f16.16x16x32.f16` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `signA`   | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `signB`   | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |
|   `b`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |
|   `c`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

### `rocdl.wmma.f16.16x16x64.bf8_bf8`(ROCDL::wmma_f16_16x16x64_bf8_bf8)

语法：

```
operation ::= `rocdl.wmma.f16.16x16x64.bf8_bf8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

### `rocdl.wmma.f16.16x16x64.bf8_fp8`(ROCDL::wmma_f16_16x16x64_bf8_fp8)

语法：

```
operation ::= `rocdl.wmma.f16.16x16x64.bf8_fp8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

### `rocdl.wmma.f16.16x16x64.fp8_bf8`(ROCDL::wmma_f16_16x16x64_fp8_bf8)

语法：

```
operation ::= `rocdl.wmma.f16.16x16x64.fp8_bf8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

### `rocdl.wmma.f16.16x16x64.fp8_fp8`(ROCDL::wmma_f16_16x16x64_fp8_fp8)

语法：

```
operation ::= `rocdl.wmma.f16.16x16x64.fp8_fp8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 16-bit float or LLVM dialect-compatible vector of 16-bit float |

### `rocdl.wmma.f32.16x16x128.bf8_bf8`(ROCDL::wmma_f32_16x16x128_bf8_bf8)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x128.bf8_bf8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.f32.16x16x128.bf8_fp8`(ROCDL::wmma_f32_16x16x128_bf8_fp8)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x128.bf8_fp8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.f32.16x16x128.fp8_bf8`(ROCDL::wmma_f32_16x16x128_fp8_bf8)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x128.fp8_bf8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.f32.16x16x128.fp8_fp8`(ROCDL::wmma_f32_16x16x128_fp8_fp8)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x128.fp8_fp8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.f32.16x16x16.bf16`(ROCDL::wmma_f32_16x16x16_bf16)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x16.bf16` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.f32.16x16x16.bf8_bf8`(ROCDL::wmma_f32_16x16x16_bf8_bf8)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x16.bf8_bf8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.f32.16x16x16.bf8_fp8`(ROCDL::wmma_f32_16x16x16_bf8_fp8)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x16.bf8_fp8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.f32.16x16x16.f16`(ROCDL::wmma_f32_16x16x16_f16)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x16.f16` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |
|   `b`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.f32.16x16x16.fp8_bf8`(ROCDL::wmma_f32_16x16x16_fp8_bf8)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x16.fp8_bf8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.f32.16x16x16.fp8_fp8`(ROCDL::wmma_f32_16x16x16_fp8_fp8)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x16.fp8_fp8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.f32.16x16x32.bf16`(ROCDL::wmma_f32_16x16x32_bf16)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x32.bf16` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `signA`   | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `signB`   | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | bfloat16 type or LLVM dialect-compatible vector of bfloat16 type |
|   `b`   | bfloat16 type or LLVM dialect-compatible vector of bfloat16 type |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.f32.16x16x32.f16`(ROCDL::wmma_f32_16x16x32_f16)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x32.f16` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `signA`   | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `signB`   | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |
|   `b`   | 16-bit float or LLVM dialect-compatible vector of 16-bit float |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.f32.16x16x4.f32`(ROCDL::wmma_f32_16x16x4_f32)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x4.f32` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `signA`   | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `signB`   | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |
|   `b`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.f32.16x16x64.bf8_bf8`(ROCDL::wmma_f32_16x16x64_bf8_bf8)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x64.bf8_bf8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.f32.16x16x64.bf8_fp8`(ROCDL::wmma_f32_16x16x64_bf8_fp8)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x64.bf8_fp8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.f32.16x16x64.fp8_bf8`(ROCDL::wmma_f32_16x16x64_fp8_bf8)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x64.fp8_bf8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.f32.16x16x64.fp8_fp8`(ROCDL::wmma_f32_16x16x64_fp8_fp8)

语法：

```
operation ::= `rocdl.wmma.f32.16x16x64.fp8_fp8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `modC`    | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | integer or LLVM dialect-compatible vector of integer         |
|   `b`   | integer or LLVM dialect-compatible vector of integer         |
|   `c`   | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.i32.16x16x16.iu4`(ROCDL::wmma_i32_16x16x16_iu4)

语法：

```
operation ::= `rocdl.wmma.i32.16x16x16.iu4` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                      |
| --------- | ------------------- | -------------------------------- |
| `signA`   | ::mlir::IntegerAttr | 1-bit signless integer attribute |
| `signB`   | ::mlir::IntegerAttr | 1-bit signless integer attribute |
| `clamp`   | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description                                          |
| :-----: | ---------------------------------------------------- |
|   `a`   | integer or LLVM dialect-compatible vector of integer |
|   `b`   | integer or LLVM dialect-compatible vector of integer |
|   `c`   | integer or LLVM dialect-compatible vector of integer |

#### 结果：

| Result | Description                                          |
| :----: | ---------------------------------------------------- |
| `res`  | integer or LLVM dialect-compatible vector of integer |

### `rocdl.wmma.i32.16x16x16.iu8`(ROCDL::wmma_i32_16x16x16_iu8)

语法：

```
operation ::= `rocdl.wmma.i32.16x16x16.iu8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                      |
| --------- | ------------------- | -------------------------------- |
| `signA`   | ::mlir::IntegerAttr | 1-bit signless integer attribute |
| `signB`   | ::mlir::IntegerAttr | 1-bit signless integer attribute |
| `clamp`   | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description                                          |
| :-----: | ---------------------------------------------------- |
|   `a`   | integer or LLVM dialect-compatible vector of integer |
|   `b`   | integer or LLVM dialect-compatible vector of integer |
|   `c`   | integer or LLVM dialect-compatible vector of integer |

#### 结果：

| Result | Description                                          |
| :----: | ---------------------------------------------------- |
| `res`  | integer or LLVM dialect-compatible vector of integer |

### `rocdl.wmma.i32.16x16x32.iu4`(ROCDL::wmma_i32_16x16x32_iu4)

语法：

```
operation ::= `rocdl.wmma.i32.16x16x32.iu4` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                      |
| --------- | ------------------- | -------------------------------- |
| `signA`   | ::mlir::IntegerAttr | 1-bit signless integer attribute |
| `signB`   | ::mlir::IntegerAttr | 1-bit signless integer attribute |
| `clamp`   | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description                                          |
| :-----: | ---------------------------------------------------- |
|   `a`   | integer or LLVM dialect-compatible vector of integer |
|   `b`   | integer or LLVM dialect-compatible vector of integer |
|   `c`   | integer or LLVM dialect-compatible vector of integer |

#### 结果：

| Result | Description                                          |
| :----: | ---------------------------------------------------- |
| `res`  | integer or LLVM dialect-compatible vector of integer |

### `rocdl.wmma.i32.16x16x64.iu8`(ROCDL::wmma_i32_16x16x64_iu8)

语法：

```
operation ::= `rocdl.wmma.i32.16x16x64.iu8` $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute | MLIR Type           | Description                      |
| --------- | ------------------- | -------------------------------- |
| `signA`   | ::mlir::IntegerAttr | 1-bit signless integer attribute |
| `signB`   | ::mlir::IntegerAttr | 1-bit signless integer attribute |
| `reuseA`  | ::mlir::IntegerAttr | 1-bit signless integer attribute |
| `reuseB`  | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description                                          |
| :-----: | ---------------------------------------------------- |
|   `a`   | integer or LLVM dialect-compatible vector of integer |
|   `b`   | integer or LLVM dialect-compatible vector of integer |
|   `c`   | integer or LLVM dialect-compatible vector of integer |

#### 结果：

| Result | Description                                          |
| :----: | ---------------------------------------------------- |
| `res`  | integer or LLVM dialect-compatible vector of integer |

### `rocdl.wmma.scale.f32.16x16x128.f8f6f4`(ROCDL::wmma_scale_f32_16x16x128_f8f6f4)

语法：

```
operation ::= `rocdl.wmma.scale.f32.16x16x128.f8f6f4` $a `,` $b `,` $c `,` $scaleA `,` $scaleB attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute    | MLIR Type           | Description                       |
| ------------ | ------------------- | --------------------------------- |
| `fmtA`       | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `fmtB`       | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `modC`       | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `scaleAType` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `fmtScaleA`  | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `scaleBType` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `fmtScaleB`  | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `reuseA`     | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`     | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand  | Description                                                  |
| :------: | ------------------------------------------------------------ |
|   `a`    | integer or LLVM dialect-compatible vector of integer         |
|   `b`    | integer or LLVM dialect-compatible vector of integer         |
|   `c`    | 32-bit float or LLVM dialect-compatible vector of 32-bit float |
| `scaleA` | 32-bit signless integer                                      |
| `scaleB` | 32-bit signless integer                                      |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.scale.f32.32x16x128.f4`(ROCDL::wmma_scale_f32_32x16x128_f4)

语法：

```
operation ::= `rocdl.wmma.scale.f32.32x16x128.f4` $a `,` $b `,` $c `,` $scaleA `,` $scaleB attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute    | MLIR Type           | Description                       |
| ------------ | ------------------- | --------------------------------- |
| `modC`       | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `scaleAType` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `fmtScaleA`  | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `scaleBType` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `fmtScaleB`  | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `reuseA`     | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`     | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand  | Description                                                  |
| :------: | ------------------------------------------------------------ |
|   `a`    | integer or LLVM dialect-compatible vector of integer         |
|   `b`    | integer or LLVM dialect-compatible vector of integer         |
|   `c`    | 32-bit float or LLVM dialect-compatible vector of 32-bit float |
| `scaleA` | 32-bit signless integer                                      |
| `scaleB` | 32-bit signless integer                                      |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.scale16.f32.16x16x128.f8f6f4`(ROCDL::wmma_scale16_f32_16x16x128_f8f6f4)

语法：

```
operation ::= `rocdl.wmma.scale16.f32.16x16x128.f8f6f4` $a `,` $b `,` $c `,` $scaleA `,` $scaleB attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute    | MLIR Type           | Description                       |
| ------------ | ------------------- | --------------------------------- |
| `fmtA`       | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `fmtB`       | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `modC`       | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `scaleAType` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `fmtScaleA`  | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `scaleBType` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `fmtScaleB`  | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `reuseA`     | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`     | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand  | Description                                                  |
| :------: | ------------------------------------------------------------ |
|   `a`    | integer or LLVM dialect-compatible vector of integer         |
|   `b`    | integer or LLVM dialect-compatible vector of integer         |
|   `c`    | 32-bit float or LLVM dialect-compatible vector of 32-bit float |
| `scaleA` | 64-bit signless integer                                      |
| `scaleB` | 64-bit signless integer                                      |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.wmma.scale16.f32.32x16x128.f4`(ROCDL::wmma_scale16_f32_32x16x128_f4)

语法：

```
operation ::= `rocdl.wmma.scale16.f32.32x16x128.f4` $a `,` $b `,` $c `,` $scaleA `,` $scaleB attr-dict `:` functional-type(operands, $res)
```

#### 属性：

| Attribute    | MLIR Type           | Description                       |
| ------------ | ------------------- | --------------------------------- |
| `modC`       | ::mlir::IntegerAttr | 16-bit signless integer attribute |
| `scaleAType` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `fmtScaleA`  | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `scaleBType` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `fmtScaleB`  | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `reuseA`     | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `reuseB`     | ::mlir::IntegerAttr | 1-bit signless integer attribute  |

#### 操作数：

| Operand  | Description                                                  |
| :------: | ------------------------------------------------------------ |
|   `a`    | integer or LLVM dialect-compatible vector of integer         |
|   `b`    | integer or LLVM dialect-compatible vector of integer         |
|   `c`    | 32-bit float or LLVM dialect-compatible vector of 32-bit float |
| `scaleA` | 64-bit signless integer                                      |
| `scaleB` | 64-bit signless integer                                      |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 32-bit float or LLVM dialect-compatible vector of 32-bit float |

### `rocdl.workgroup.dim.x`(ROCDL::BlockDimXOp)

语法：

```
operation ::= `rocdl.workgroup.dim.x` (`range` $range^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                                                  |
| --------- | ------------------------------- | ------------------------------------------------------------ |
| `range`   | ::mlir::LLVM::ConstantRangeAttr | A range of two integers, corresponding to LLVM's ConstantRange`A pair of two integers, mapping to the ConstantRange structure in LLVM IR, which is allowed to wrap or be empty. The range represented is [Lower, Upper), and is either signed or unsigned depending on context. lower and upper must have the same width. Syntax: `<` `i`(width($lower)) $lower `,` $upper `>` ` |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.workgroup.dim.y`(ROCDL::BlockDimYOp)

语法：

```
operation ::= `rocdl.workgroup.dim.y` (`range` $range^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                                                  |
| --------- | ------------------------------- | ------------------------------------------------------------ |
| `range`   | ::mlir::LLVM::ConstantRangeAttr | A range of two integers, corresponding to LLVM's ConstantRange`A pair of two integers, mapping to the ConstantRange structure in LLVM IR, which is allowed to wrap or be empty. The range represented is [Lower, Upper), and is either signed or unsigned depending on context. lower and upper must have the same width. Syntax: `<` `i`(width($lower)) $lower `,` $upper `>` ` |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.workgroup.dim.z`(ROCDL::BlockDimZOp)

语法：

```
operation ::= `rocdl.workgroup.dim.z` (`range` $range^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                                                  |
| --------- | ------------------------------- | ------------------------------------------------------------ |
| `range`   | ::mlir::LLVM::ConstantRangeAttr | A range of two integers, corresponding to LLVM's ConstantRange`A pair of two integers, mapping to the ConstantRange structure in LLVM IR, which is allowed to wrap or be empty. The range represented is [Lower, Upper), and is either signed or unsigned depending on context. lower and upper must have the same width. Syntax: `<` `i`(width($lower)) $lower `,` $upper `>` ` |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.workgroup.id.x`(ROCDL::BlockIdXOp)

语法：

```
operation ::= `rocdl.workgroup.id.x` (`range` $range^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                                                  |
| --------- | ------------------------------- | ------------------------------------------------------------ |
| `range`   | ::mlir::LLVM::ConstantRangeAttr | A range of two integers, corresponding to LLVM's ConstantRange`A pair of two integers, mapping to the ConstantRange structure in LLVM IR, which is allowed to wrap or be empty. The range represented is [Lower, Upper), and is either signed or unsigned depending on context. lower and upper must have the same width. Syntax: `<` `i`(width($lower)) $lower `,` $upper `>` ` |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.workgroup.id.y`(ROCDL::BlockIdYOp)

语法：

```
operation ::= `rocdl.workgroup.id.y` (`range` $range^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                                                  |
| --------- | ------------------------------- | ------------------------------------------------------------ |
| `range`   | ::mlir::LLVM::ConstantRangeAttr | A range of two integers, corresponding to LLVM's ConstantRange`A pair of two integers, mapping to the ConstantRange structure in LLVM IR, which is allowed to wrap or be empty. The range represented is [Lower, Upper), and is either signed or unsigned depending on context. lower and upper must have the same width. Syntax: `<` `i`(width($lower)) $lower `,` $upper `>` ` |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.workgroup.id.z`(ROCDL::BlockIdZOp)

语法：

```
operation ::= `rocdl.workgroup.id.z` (`range` $range^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                                                  |
| --------- | ------------------------------- | ------------------------------------------------------------ |
| `range`   | ::mlir::LLVM::ConstantRangeAttr | A range of two integers, corresponding to LLVM's ConstantRange`A pair of two integers, mapping to the ConstantRange structure in LLVM IR, which is allowed to wrap or be empty. The range represented is [Lower, Upper), and is either signed or unsigned depending on context. lower and upper must have the same width. Syntax: `<` `i`(width($lower)) $lower `,` $upper `>` ` |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.workitem.id.x`(ROCDL::ThreadIdXOp)

语法：

```
operation ::= `rocdl.workitem.id.x` (`range` $range^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                                                  |
| --------- | ------------------------------- | ------------------------------------------------------------ |
| `range`   | ::mlir::LLVM::ConstantRangeAttr | A range of two integers, corresponding to LLVM's ConstantRange`A pair of two integers, mapping to the ConstantRange structure in LLVM IR, which is allowed to wrap or be empty. The range represented is [Lower, Upper), and is either signed or unsigned depending on context. lower and upper must have the same width. Syntax: `<` `i`(width($lower)) $lower `,` $upper `>` ` |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.workitem.id.y`(ROCDL::ThreadIdYOp)

语法：

```
operation ::= `rocdl.workitem.id.y` (`range` $range^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                                                  |
| --------- | ------------------------------- | ------------------------------------------------------------ |
| `range`   | ::mlir::LLVM::ConstantRangeAttr | A range of two integers, corresponding to LLVM's ConstantRange`A pair of two integers, mapping to the ConstantRange structure in LLVM IR, which is allowed to wrap or be empty. The range represented is [Lower, Upper), and is either signed or unsigned depending on context. lower and upper must have the same width. Syntax: `<` `i`(width($lower)) $lower `,` $upper `>` ` |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `rocdl.workitem.id.z`(ROCDL::ThreadIdZOp)

语法：

```
operation ::= `rocdl.workitem.id.z` (`range` $range^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                                                  |
| --------- | ------------------------------- | ------------------------------------------------------------ |
| `range`   | ::mlir::LLVM::ConstantRangeAttr | A range of two integers, corresponding to LLVM's ConstantRange`A pair of two integers, mapping to the ConstantRange structure in LLVM IR, which is allowed to wrap or be empty. The range represented is [Lower, Upper), and is either signed or unsigned depending on context. lower and upper must have the same width. Syntax: `<` `i`(width($lower)) $lower `,` $upper `>` ` |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

## 属性

### ROCDLTargetAttr

语法：

```
#rocdl.target<
  int,   # O
  ::llvm::StringRef,   # triple
  ::llvm::StringRef,   # chip
  ::llvm::StringRef,   # features
  ::llvm::StringRef,   # abi
  DictionaryAttr,   # flags
  ArrayAttr   # link
>
```

用于控制AMDGPU目标编译的ROCDL目标属性。如果未指定，所有参数都将使用默认值。

示例：

1. 采用默认值的目标。

```
  gpu.module @mymodule [#rocdl.target] attributes {...} {
    ...
  }
```

2. 使用`gfx90a`芯片和快速数学操作的目标。

```
  gpu.module @mymodule [#rocdl.target<chip = "gfx90a", flags = {fast, no_wave64}>] {
    ...
  }
```

#### 参数：

| Parameter |      C++ type       | Description                       |
| :-------: | :-----------------: | --------------------------------- |
|     O     |        `int`        | Optimization level to apply.      |
|  triple   | `::llvm::StringRef` | Target triple.                    |
|   chip    | `::llvm::StringRef` | Target chip.                      |
| features  | `::llvm::StringRef` | Target chip features.             |
|    abi    | `::llvm::StringRef` | ABI version.                      |
|   flags   |  `DictionaryAttr`   | Target specific flags.            |
|   link    |     `ArrayAttr`     | Files to link to the LLVM module. |