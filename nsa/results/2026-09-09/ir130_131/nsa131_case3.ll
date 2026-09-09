; ModuleID = '/data/nsa_codex_20260909_r1/results/ir130_131/nsa131_case3.bc'
source_filename = "ld-temp.o"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "mxc-metax-macahca"

%struct.mcDevMallocInfo.0 = type { i32, i32, ptr }
%struct.__half = type { i16 }

@buf_dyn_shmem = external protected local_unnamed_addr addrspace(3) global [0 x i8], align 1024
@mcDeviceMemoryInfo = weak protected addrspace(1) externally_initialized global [1 x %struct.mcDevMallocInfo.0] zeroinitializer, align 8
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @mcDeviceMemoryInfo to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress noreturn nounwind
define weak void @__cxa_pure_virtual() local_unnamed_addr #0 {
  tail call void @llvm.trap()
  unreachable
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #1

; Function Attrs: mustprogress noreturn nounwind
define weak void @__cxa_deleted_virtual() local_unnamed_addr #0 {
  tail call void @llvm.trap()
  unreachable
}

; Function Attrs: nounwind
define weak protected void @__mcImplicitDeviceSynchronize() local_unnamed_addr #2 {
  %1 = tail call ptr addrspace(4) @llvm.mxc.implicitarg.ptr()
  %2 = getelementptr inbounds i8, ptr addrspace(4) %1, i64 32
  %3 = load i64, ptr addrspace(4) %2, align 8, !tbaa !3
  %4 = inttoptr i64 %3 to ptr
  %5 = getelementptr inbounds i8, ptr addrspace(4) %1, i64 40
  %6 = load i64, ptr addrspace(4) %5, align 8, !tbaa !3
  %7 = inttoptr i64 %6 to ptr
  %8 = getelementptr inbounds i8, ptr %7, i64 12
  %9 = getelementptr inbounds i8, ptr %7, i64 16
  %10 = load ptr, ptr %9, align 8, !tbaa !7
  %11 = icmp eq ptr %10, null
  br i1 %11, label %51, label %12

12:                                               ; preds = %0
  %13 = tail call align 4 dereferenceable(64) ptr addrspace(4) @llvm.mxc.dispatch.ptr()
  %14 = getelementptr inbounds i8, ptr addrspace(4) %13, i64 12
  %15 = load i32, ptr addrspace(4) %14, align 4, !range !14, !invariant.load !15
  %16 = getelementptr inbounds i8, ptr addrspace(4) %13, i64 4
  %17 = load i32, ptr addrspace(4) %16, align 4, !invariant.load !15
  %18 = and i32 %17, 65535
  %19 = udiv i32 %15, %18
  %20 = mul i32 %19, %18
  %21 = icmp ugt i32 %15, %20
  %22 = zext i1 %21 to i32
  %23 = add nuw i32 %19, %22
  %24 = getelementptr inbounds i8, ptr addrspace(4) %13, i64 16
  %25 = load i32, ptr addrspace(4) %24, align 4, !range !14, !invariant.load !15
  %26 = lshr i32 %17, 16
  %27 = udiv i32 %25, %26
  %28 = mul i32 %27, %26
  %29 = icmp ugt i32 %25, %28
  %30 = zext i1 %29 to i32
  %31 = add nuw i32 %27, %30
  %32 = tail call noundef range(i32 0, 2147483647) i32 @llvm.mxc.block.id.z(), !range !16
  %33 = mul i32 %31, %32
  %34 = tail call noundef range(i32 0, 2147483647) i32 @llvm.mxc.block.id.y(), !range !16
  %35 = add i32 %33, %34
  %36 = mul i32 %35, %23
  %37 = tail call noundef range(i32 0, 2147483647) i32 @llvm.mxc.block.id.x(), !range !16
  %38 = add i32 %36, %37
  %39 = zext i32 %38 to i64
  %40 = getelementptr inbounds i32, ptr %10, i64 %39
  %41 = tail call i1 @llvm.mxc.is.private(ptr nonnull %40)
  br label %42

42:                                               ; preds = %50, %12
  br i1 %41, label %43, label %45

43:                                               ; preds = %42
  %44 = load i32, ptr %40, align 4, !tbaa !17
  br label %47

45:                                               ; preds = %42
  %46 = atomicrmw or ptr %40, i32 0 syncscope("device-one-as") monotonic, align 4
  br label %47

47:                                               ; preds = %45, %43
  %48 = phi i32 [ %44, %43 ], [ %46, %45 ]
  %49 = icmp sgt i32 %48, 0
  br i1 %49, label %50, label %51

50:                                               ; preds = %47
  tail call void @llvm.mxc.sleep(i32 1)
  br label %42, !llvm.loop !18

51:                                               ; preds = %47, %0
  %52 = tail call i1 @llvm.mxc.is.private(ptr nonnull %8)
  br i1 %52, label %53, label %56

53:                                               ; preds = %51
  %54 = load i32, ptr %8, align 4, !tbaa !17
  %55 = add nsw i32 %54, -1
  store i32 %55, ptr %8, align 4, !tbaa !17
  br label %58

56:                                               ; preds = %51
  %57 = atomicrmw add ptr %8, i32 -1 syncscope("device-one-as") monotonic, align 4
  br label %58

58:                                               ; preds = %56, %53
  %59 = phi i32 [ %54, %53 ], [ %57, %56 ]
  fence syncscope("device") seq_cst
  %60 = getelementptr inbounds i8, ptr %7, i64 4
  %61 = load i32, ptr %60, align 4, !tbaa !20
  %62 = icmp eq i32 %61, 0
  br i1 %62, label %63, label %66

63:                                               ; preds = %58
  %64 = icmp slt i32 %59, 2
  br i1 %64, label %65, label %99

65:                                               ; preds = %63
  tail call void @llvm.mxc.sleep(i32 30)
  br label %99

66:                                               ; preds = %58
  %67 = getelementptr inbounds i8, ptr %7, i64 24
  %68 = load ptr, ptr %67, align 8, !tbaa !21
  %69 = load i32, ptr %7, align 8, !tbaa !22
  %70 = getelementptr inbounds i8, ptr %68, i64 16
  %71 = load ptr, ptr %70, align 8, !tbaa !7
  %72 = zext i32 %69 to i64
  %73 = getelementptr inbounds i32, ptr %71, i64 %72
  %74 = icmp slt i32 %59, 2
  br i1 %74, label %75, label %92

75:                                               ; preds = %66
  %76 = getelementptr inbounds i8, ptr %4, i64 24
  %77 = load ptr, ptr %76, align 8, !tbaa !23
  %78 = ptrtoint ptr %77 to i64
  %79 = sub i64 %6, %78
  %80 = sdiv exact i64 %79, 96
  %81 = getelementptr inbounds i8, ptr %4, i64 80
  %82 = load i64, ptr %81, align 8, !tbaa !25
  %83 = inttoptr i64 %82 to ptr
  %84 = and i64 %80, 4294967295
  %85 = getelementptr inbounds i32, ptr %83, i64 %84
  %86 = tail call i1 @llvm.mxc.is.private(ptr %85)
  br i1 %86, label %87, label %90

87:                                               ; preds = %75
  %88 = load i32, ptr %85, align 4, !tbaa !17
  %89 = and i32 %88, 2147483647
  store i32 %89, ptr %85, align 4, !tbaa !17
  br label %92

90:                                               ; preds = %75
  %91 = atomicrmw and ptr %85, i32 2147483647 syncscope("device-one-as") monotonic, align 4
  br label %92

92:                                               ; preds = %90, %87, %66
  %93 = tail call i1 @llvm.mxc.is.private(ptr %73)
  br i1 %93, label %94, label %97

94:                                               ; preds = %92
  %95 = load i32, ptr %73, align 4, !tbaa !17
  %96 = add nsw i32 %95, -1
  store i32 %96, ptr %73, align 4, !tbaa !17
  br label %99

97:                                               ; preds = %92
  %98 = atomicrmw add ptr %73, i32 -1 syncscope("device-one-as") monotonic, align 4
  br label %99

99:                                               ; preds = %97, %94, %65, %63
  ret void
}

; Function Attrs: nounwind speculatable willreturn memory(none)
declare align 4 ptr addrspace(4) @llvm.mxc.implicitarg.ptr() #3

; Function Attrs: nounwind speculatable willreturn memory(none)
declare align 4 ptr addrspace(4) @llvm.mxc.dispatch.ptr() #3

; Function Attrs: nounwind speculatable willreturn memory(none)
declare i32 @llvm.mxc.block.id.z() #3

; Function Attrs: nounwind speculatable willreturn memory(none)
declare i32 @llvm.mxc.block.id.y() #3

; Function Attrs: nounwind speculatable willreturn memory(none)
declare i32 @llvm.mxc.block.id.x() #3

; Function Attrs: nounwind speculatable willreturn memory(none)
declare i1 @llvm.mxc.is.private(ptr nocapture) #3

; Function Attrs: nounwind willreturn
declare void @llvm.mxc.sleep(i32 immarg) #4

; Function Attrs: convergent mustprogress norecurse nounwind willreturn
define protected metaxgpu_kernel void @kernel_kernel(ptr addrspace(1) noalias nocapture noundef readonly %0, ptr addrspace(4) noalias nocapture noundef readonly %1, ptr addrspace(1) noalias nocapture noundef writeonly %2, ptr addrspace(4) noalias nocapture noundef readonly %3, ptr addrspace(4) noalias nocapture noundef readonly %4) local_unnamed_addr #5 {
  %6 = tail call noundef range(i32 0, 2147483647) i32 @llvm.mxc.block.id.x(), !range !16
  %7 = zext nneg i32 %6 to i64
  %8 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %7
  %9 = load i32, ptr addrspace(1) %8, align 4, !tbaa !17
  %10 = lshr i32 %6, 4
  %11 = icmp ugt i32 %9, %10
  br i1 %11, label %671, label %12

12:                                               ; preds = %5
  %13 = shl nuw nsw i32 %9, 4
  %14 = and i32 %13, 240
  %15 = tail call noundef range(i32 0, 1024) i32 @llvm.mxc.thread.id.x(), !range !26
  %16 = lshr i32 %15, 2
  %17 = and i32 %16, 252
  %18 = add nuw nsw i32 %14, %17
  %19 = icmp ugt i32 %18, %6
  %20 = select i1 %19, float 0xFFF0000000000000, float 0.000000e+00
  %21 = insertelement <4 x float> poison, float %20, i64 0
  %22 = icmp ult i32 %18, %6
  %23 = select i1 %22, float 0.000000e+00, float 0xFFF0000000000000
  %24 = insertelement <4 x float> %21, float %23, i64 1
  %25 = or disjoint i32 %18, 2
  %26 = icmp ugt i32 %25, %6
  %27 = select i1 %26, float 0xFFF0000000000000, float 0.000000e+00
  %28 = insertelement <4 x float> %24, float %27, i64 2
  %29 = or disjoint i32 %18, 3
  %30 = icmp ugt i32 %29, %6
  %31 = select i1 %30, float 0xFFF0000000000000, float 0.000000e+00
  %32 = insertelement <4 x float> %28, float %31, i64 3
  %33 = shl nsw i32 %6, 11
  %34 = shl nuw nsw i32 %15, 4
  %35 = and i32 %34, 16256
  %36 = add nuw nsw i32 %35, %33
  %37 = shl nuw nsw i32 %15, 3
  %38 = and i32 %37, 56
  %39 = or disjoint i32 %36, %38
  %40 = and i32 %37, 8128
  %41 = and i32 %37, 32
  %42 = add nuw nsw i32 %41, %15
  %43 = and i32 %42, 32
  %44 = and i32 %37, 16
  %45 = add nuw nsw i32 %44, %15
  %46 = and i32 %45, 16
  %47 = mul nuw nsw i32 %15, 9
  %48 = and i32 %47, 8
  %49 = shl i32 %9, 11
  %50 = and i32 %49, 30720
  %51 = or disjoint i32 %38, %35
  %52 = add nuw nsw i32 %51, %50
  %53 = shl nuw nsw i32 %15, 6
  %54 = and i32 %53, 960
  %55 = lshr i32 %15, 1
  %56 = lshr i32 %15, 5
  %57 = add nuw nsw i32 %56, %15
  %58 = shl nuw nsw i32 %57, 3
  %59 = and i32 %58, 8
  %60 = and i32 %16, 4
  %61 = or disjoint i32 %60, %54
  %62 = or disjoint i32 %61, %59
  %63 = zext nneg i32 %39 to i64
  %64 = zext nneg i32 %52 to i64
  %65 = getelementptr inbounds %struct.__half, ptr addrspace(4) %3, i64 %63
  %66 = or disjoint i32 %40, %43
  %67 = or disjoint i32 %66, %46
  %68 = or disjoint i32 %67, %48
  %69 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 2048), i32 %68
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %69, ptr addrspace(4) noundef align 16 dereferenceable(16) %65, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %70 = getelementptr inbounds i8, ptr addrspace(4) %65, i64 2048
  %71 = add nuw nsw i32 %40, 512
  %72 = or disjoint i32 %71, %43
  %73 = or disjoint i32 %72, %46
  %74 = or disjoint i32 %73, %48
  %75 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 2048), i32 %74
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %75, ptr addrspace(4) noundef align 16 dereferenceable(16) %70, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %76 = getelementptr inbounds %struct.__half, ptr addrspace(4) %1, i64 %64
  %77 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %68
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %77, ptr addrspace(4) noundef align 16 dereferenceable(16) %76, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  %78 = getelementptr inbounds i8, ptr addrspace(4) %76, i64 2048
  %79 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %74
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %79, ptr addrspace(4) noundef align 16 dereferenceable(16) %78, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  %80 = shl nuw nsw i32 %16, 5
  %81 = and i32 %80, 32
  %82 = shl nuw nsw i32 %55, 4
  %83 = and i32 %82, 16
  %84 = or disjoint i32 %62, %83
  %85 = or disjoint i32 %84, %81
  %86 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 2048), i32 %85
  %87 = load <4 x half>, ptr addrspace(3) %86, align 8
  %88 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %85
  %89 = load <4 x half>, ptr addrspace(3) %88, align 8
  %90 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %89, <4 x half> %87, <4 x float> %32)
  %91 = xor i32 %85, 16
  %92 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 2048), i32 %91
  %93 = load <4 x half>, ptr addrspace(3) %92, align 8
  %94 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %91
  %95 = load <4 x half>, ptr addrspace(3) %94, align 8
  %96 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %95, <4 x half> %93, <4 x float> %90)
  %97 = or disjoint i32 %81, %84
  %98 = xor i32 %97, 32
  %99 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 2048), i32 %98
  %100 = load <4 x half>, ptr addrspace(3) %99, align 8
  %101 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %98
  %102 = load <4 x half>, ptr addrspace(3) %101, align 8
  %103 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %102, <4 x half> %100, <4 x float> %96)
  %104 = xor i32 %97, 48
  %105 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 2048), i32 %104
  %106 = load <4 x half>, ptr addrspace(3) %105, align 8
  %107 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %104
  %108 = load <4 x half>, ptr addrspace(3) %107, align 8
  %109 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %108, <4 x half> %106, <4 x float> %103)
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  %110 = or disjoint i64 %63, 64
  %111 = getelementptr inbounds %struct.__half, ptr addrspace(4) %3, i64 %110
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %69, ptr addrspace(4) noundef align 16 dereferenceable(16) %111, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %112 = getelementptr inbounds i8, ptr addrspace(4) %65, i64 2176
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %75, ptr addrspace(4) noundef align 16 dereferenceable(16) %112, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %113 = or disjoint i64 %64, 64
  %114 = getelementptr inbounds %struct.__half, ptr addrspace(4) %1, i64 %113
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %77, ptr addrspace(4) noundef align 16 dereferenceable(16) %114, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  %115 = getelementptr inbounds i8, ptr addrspace(4) %76, i64 2176
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %79, ptr addrspace(4) noundef align 16 dereferenceable(16) %115, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  %116 = load <4 x half>, ptr addrspace(3) %86, align 8
  %117 = load <4 x half>, ptr addrspace(3) %88, align 8
  %118 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %117, <4 x half> %116, <4 x float> %109)
  %119 = load <4 x half>, ptr addrspace(3) %92, align 8
  %120 = load <4 x half>, ptr addrspace(3) %94, align 8
  %121 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %120, <4 x half> %119, <4 x float> %118)
  %122 = load <4 x half>, ptr addrspace(3) %99, align 8
  %123 = load <4 x half>, ptr addrspace(3) %101, align 8
  %124 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %123, <4 x half> %122, <4 x float> %121)
  %125 = load <4 x half>, ptr addrspace(3) %105, align 8
  %126 = load <4 x half>, ptr addrspace(3) %107, align 8
  %127 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %126, <4 x half> %125, <4 x float> %124)
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  %128 = extractelement <4 x float> %127, i64 0
  %129 = tail call contract noundef float @llvm.maxnum.f32(float %128, float 0xFFF0000000000000)
  %130 = extractelement <4 x float> %127, i64 1
  %131 = tail call contract noundef float @llvm.maxnum.f32(float %129, float %130)
  %132 = extractelement <4 x float> %127, i64 2
  %133 = tail call contract noundef float @llvm.maxnum.f32(float %131, float %132)
  %134 = extractelement <4 x float> %127, i64 3
  %135 = tail call contract noundef float @llvm.maxnum.f32(float %133, float %134)
  %136 = bitcast float %135 to i32
  %137 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #11
  %138 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %137) #11
  %139 = xor i32 %138, 32
  %140 = and i32 %138, -64
  %141 = add nsw i32 %140, 64
  %142 = icmp slt i32 %139, %141
  %143 = select i1 %142, i32 %139, i32 %138
  %144 = shl i32 %143, 2
  %145 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %144, i32 %136)
  %146 = bitcast i32 %145 to float
  %147 = tail call contract noundef float @llvm.maxnum.f32(float %135, float %146)
  %148 = bitcast float %147 to i32
  %149 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #11
  %150 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %149) #11
  %151 = xor i32 %150, 16
  %152 = and i32 %150, -64
  %153 = add nsw i32 %152, 64
  %154 = icmp slt i32 %151, %153
  %155 = select i1 %154, i32 %151, i32 %150
  %156 = shl i32 %155, 2
  %157 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %156, i32 %148)
  %158 = bitcast i32 %157 to float
  %159 = tail call contract noundef float @llvm.maxnum.f32(float %147, float %158)
  %160 = fsub contract float %128, %159
  %161 = fmul contract float %160, 0x3FC0527DC0000000
  %162 = fcmp contract olt float %161, -1.260000e+02
  %163 = select contract i1 %162, float 6.400000e+01, float 0.000000e+00
  %164 = fadd contract float %161, %163
  %165 = tail call contract float @llvm.exp2.f32(float %164)
  %166 = select contract i1 %162, float 0x3BF0000000000000, float 1.000000e+00
  %167 = fmul contract float %166, %165
  %168 = fsub contract float %130, %159
  %169 = fmul contract float %168, 0x3FC0527DC0000000
  %170 = fcmp contract olt float %169, -1.260000e+02
  %171 = select contract i1 %170, float 6.400000e+01, float 0.000000e+00
  %172 = fadd contract float %169, %171
  %173 = tail call contract float @llvm.exp2.f32(float %172)
  %174 = select contract i1 %170, float 0x3BF0000000000000, float 1.000000e+00
  %175 = fmul contract float %174, %173
  %176 = fsub contract float %132, %159
  %177 = fmul contract float %176, 0x3FC0527DC0000000
  %178 = fcmp contract olt float %177, -1.260000e+02
  %179 = select contract i1 %178, float 6.400000e+01, float 0.000000e+00
  %180 = fadd contract float %177, %179
  %181 = tail call contract float @llvm.exp2.f32(float %180)
  %182 = select contract i1 %178, float 0x3BF0000000000000, float 1.000000e+00
  %183 = fmul contract float %182, %181
  %184 = fsub contract float %134, %159
  %185 = fmul contract float %184, 0x3FC0527DC0000000
  %186 = fcmp contract olt float %185, -1.260000e+02
  %187 = select contract i1 %186, float 6.400000e+01, float 0.000000e+00
  %188 = fadd contract float %185, %187
  %189 = tail call contract float @llvm.exp2.f32(float %188)
  %190 = select contract i1 %186, float 0x3BF0000000000000, float 1.000000e+00
  %191 = fmul contract float %190, %189
  %192 = fadd contract float %167, 0.000000e+00
  %193 = fadd contract float %192, %175
  %194 = fadd contract float %193, %183
  %195 = fadd contract float %194, %191
  %196 = bitcast float %195 to i32
  %197 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #11
  %198 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %197) #11
  %199 = xor i32 %198, 32
  %200 = and i32 %198, -64
  %201 = add nsw i32 %200, 64
  %202 = icmp slt i32 %199, %201
  %203 = select i1 %202, i32 %199, i32 %198
  %204 = shl i32 %203, 2
  %205 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %204, i32 %196)
  %206 = bitcast i32 %205 to float
  %207 = fadd contract float %195, %206
  %208 = bitcast float %207 to i32
  %209 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #11
  %210 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %209) #11
  %211 = xor i32 %210, 16
  %212 = and i32 %210, -64
  %213 = add nsw i32 %212, 64
  %214 = icmp slt i32 %211, %213
  %215 = select i1 %214, i32 %211, i32 %210
  %216 = shl i32 %215, 2
  %217 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %216, i32 %208)
  %218 = bitcast i32 %217 to float
  %219 = fadd contract float %207, %218
  %220 = fdiv contract float %167, %219
  %221 = fdiv contract float %175, %219
  %222 = fdiv contract float %183, %219
  %223 = fdiv contract float %191, %219
  %224 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !30
  %225 = fptrunc float %220 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %224), !noalias !30
  %226 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !30
  %227 = fptrunc float %221 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %226), !noalias !30
  %228 = insertelement <4 x half> poison, half %225, i64 0
  %229 = insertelement <4 x half> %228, half %227, i64 1
  %230 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !35
  %231 = fptrunc float %222 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %230), !noalias !35
  %232 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !35
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %232), !noalias !35
  %233 = insertelement <4 x half> %229, half %231, i64 2
  %234 = add nuw nsw i32 %50, %37
  %235 = shl nuw nsw i32 %15, 7
  %236 = and i32 %235, 1024
  %237 = shl nuw nsw i32 %15, 2
  %238 = and i32 %237, 4032
  %239 = add nuw nsw i32 %238, %236
  %240 = add nuw nsw i32 %56, %55
  %241 = shl nuw nsw i32 %240, 4
  %242 = and i32 %241, 16
  %243 = lshr i32 %15, 4
  %244 = add nuw nsw i32 %243, %15
  %245 = shl nuw nsw i32 %244, 3
  %246 = and i32 %245, 8
  %247 = or disjoint i32 %239, %242
  %248 = or disjoint i32 %247, %246
  %249 = zext nneg i32 %234 to i64
  %250 = getelementptr inbounds %struct.__half, ptr addrspace(4) %4, i64 %249
  %251 = or disjoint i32 %248, %81
  %252 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %251
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %252, ptr addrspace(4) noundef align 16 dereferenceable(16) %250, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !40
  %253 = getelementptr inbounds i8, ptr addrspace(4) %250, i64 1024
  %254 = add nuw nsw i32 %248, 256
  %255 = or disjoint i32 %81, %254
  %256 = xor i32 %255, 32
  %257 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %256
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %257, ptr addrspace(4) noundef align 16 dereferenceable(16) %253, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !40
  %258 = getelementptr inbounds i8, ptr addrspace(4) %250, i64 2048
  %259 = add nuw nsw i32 %248, 512
  %260 = or disjoint i32 %259, %81
  %261 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %260
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %261, ptr addrspace(4) noundef align 16 dereferenceable(16) %258, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !40
  %262 = getelementptr inbounds i8, ptr addrspace(4) %250, i64 3072
  %263 = add nuw nsw i32 %248, 768
  %264 = or disjoint i32 %81, %263
  %265 = xor i32 %264, 32
  %266 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %265
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %266, ptr addrspace(4) noundef align 16 dereferenceable(16) %262, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !40
  %267 = fptrunc float %223 to half
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  %268 = and i32 %34, 16128
  %269 = and i32 %15, 7
  %270 = shl nuw nsw i32 %243, 5
  %271 = and i32 %270, 32
  %272 = or disjoint i32 %268, %271
  %273 = and i32 %15, 8
  %274 = or disjoint i32 %272, %273
  %275 = or disjoint i32 %274, %269
  %276 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %275
  %277 = load half, ptr addrspace(3) %276, align 2, !tbaa !41
  %278 = insertelement <4 x half> poison, half %277, i64 0
  %279 = or disjoint i32 %268, 64
  %280 = or disjoint i32 %279, %271
  %281 = xor i32 %273, 8
  %282 = or disjoint i32 %280, %281
  %283 = or disjoint i32 %282, %269
  %284 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %283
  %285 = load half, ptr addrspace(3) %284, align 2, !tbaa !41
  %286 = insertelement <4 x half> %278, half %285, i64 1
  %287 = or disjoint i32 %268, 128
  %288 = or disjoint i32 %287, %271
  %289 = or disjoint i32 %288, %273
  %290 = or disjoint i32 %289, %269
  %291 = or disjoint i32 %290, 16
  %292 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %291
  %293 = load half, ptr addrspace(3) %292, align 2, !tbaa !41
  %294 = insertelement <4 x half> %286, half %293, i64 2
  %295 = or disjoint i32 %268, 192
  %296 = or disjoint i32 %295, %271
  %297 = or disjoint i32 %296, %281
  %298 = or disjoint i32 %297, %269
  %299 = or disjoint i32 %298, 16
  %300 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %299
  %301 = load half, ptr addrspace(3) %300, align 2, !tbaa !41
  %302 = insertelement <4 x half> %294, half %301, i64 3
  %303 = or disjoint i32 %275, 16
  %304 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %303
  %305 = load half, ptr addrspace(3) %304, align 2, !tbaa !41
  %306 = insertelement <4 x half> poison, half %305, i64 0
  %307 = or disjoint i32 %283, 16
  %308 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %307
  %309 = load half, ptr addrspace(3) %308, align 2, !tbaa !41
  %310 = insertelement <4 x half> %306, half %309, i64 1
  %311 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %290
  %312 = load half, ptr addrspace(3) %311, align 2, !tbaa !41
  %313 = insertelement <4 x half> %310, half %312, i64 2
  %314 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %298
  %315 = load half, ptr addrspace(3) %314, align 2, !tbaa !41
  %316 = insertelement <4 x half> %313, half %315, i64 3
  %317 = xor i32 %271, 32
  %318 = or disjoint i32 %268, %317
  %319 = or disjoint i32 %318, %273
  %320 = or disjoint i32 %319, %269
  %321 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %320
  %322 = load half, ptr addrspace(3) %321, align 2, !tbaa !41
  %323 = insertelement <4 x half> poison, half %322, i64 0
  %324 = or disjoint i32 %279, %317
  %325 = or disjoint i32 %324, %281
  %326 = or disjoint i32 %325, %269
  %327 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %326
  %328 = load half, ptr addrspace(3) %327, align 2, !tbaa !41
  %329 = insertelement <4 x half> %323, half %328, i64 1
  %330 = or disjoint i32 %287, %317
  %331 = or disjoint i32 %330, %273
  %332 = or disjoint i32 %331, %269
  %333 = or disjoint i32 %332, 16
  %334 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %333
  %335 = load half, ptr addrspace(3) %334, align 2, !tbaa !41
  %336 = insertelement <4 x half> %329, half %335, i64 2
  %337 = or disjoint i32 %295, %317
  %338 = or disjoint i32 %337, %281
  %339 = or disjoint i32 %338, %269
  %340 = or disjoint i32 %339, 16
  %341 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %340
  %342 = load half, ptr addrspace(3) %341, align 2, !tbaa !41
  %343 = insertelement <4 x half> %336, half %342, i64 3
  %344 = or disjoint i32 %320, 16
  %345 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %344
  %346 = load half, ptr addrspace(3) %345, align 2, !tbaa !41
  %347 = insertelement <4 x half> poison, half %346, i64 0
  %348 = or disjoint i32 %326, 16
  %349 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %348
  %350 = load half, ptr addrspace(3) %349, align 2, !tbaa !41
  %351 = insertelement <4 x half> %347, half %350, i64 1
  %352 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %332
  %353 = load half, ptr addrspace(3) %352, align 2, !tbaa !41
  %354 = insertelement <4 x half> %351, half %353, i64 2
  %355 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %339
  %356 = load half, ptr addrspace(3) %355, align 2, !tbaa !41
  %357 = insertelement <4 x half> %354, half %356, i64 3
  %358 = add nuw nsw i32 %268, 1024
  %359 = or disjoint i32 %358, %271
  %360 = or disjoint i32 %359, %273
  %361 = or disjoint i32 %360, %269
  %362 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %361
  %363 = load half, ptr addrspace(3) %362, align 2, !tbaa !41
  %364 = insertelement <4 x half> poison, half %363, i64 0
  %365 = add nuw nsw i32 %268, 1088
  %366 = or disjoint i32 %365, %271
  %367 = or disjoint i32 %366, %281
  %368 = or disjoint i32 %367, %269
  %369 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %368
  %370 = load half, ptr addrspace(3) %369, align 2, !tbaa !41
  %371 = insertelement <4 x half> %364, half %370, i64 1
  %372 = add nuw nsw i32 %268, 1152
  %373 = or disjoint i32 %372, %271
  %374 = or disjoint i32 %373, %273
  %375 = or disjoint i32 %374, %269
  %376 = or disjoint i32 %375, 16
  %377 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %376
  %378 = load half, ptr addrspace(3) %377, align 2, !tbaa !41
  %379 = insertelement <4 x half> %371, half %378, i64 2
  %380 = add nuw nsw i32 %268, 1216
  %381 = or disjoint i32 %380, %271
  %382 = or disjoint i32 %381, %281
  %383 = or disjoint i32 %382, %269
  %384 = or disjoint i32 %383, 16
  %385 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %384
  %386 = load half, ptr addrspace(3) %385, align 2, !tbaa !41
  %387 = insertelement <4 x half> %379, half %386, i64 3
  %388 = or disjoint i32 %361, 16
  %389 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %388
  %390 = load half, ptr addrspace(3) %389, align 2, !tbaa !41
  %391 = insertelement <4 x half> poison, half %390, i64 0
  %392 = or disjoint i32 %368, 16
  %393 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %392
  %394 = load half, ptr addrspace(3) %393, align 2, !tbaa !41
  %395 = insertelement <4 x half> %391, half %394, i64 1
  %396 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %375
  %397 = load half, ptr addrspace(3) %396, align 2, !tbaa !41
  %398 = insertelement <4 x half> %395, half %397, i64 2
  %399 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %383
  %400 = load half, ptr addrspace(3) %399, align 2, !tbaa !41
  %401 = insertelement <4 x half> %398, half %400, i64 3
  %402 = or disjoint i32 %358, %317
  %403 = or disjoint i32 %402, %273
  %404 = or disjoint i32 %403, %269
  %405 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %404
  %406 = load half, ptr addrspace(3) %405, align 2, !tbaa !41
  %407 = insertelement <4 x half> poison, half %406, i64 0
  %408 = or disjoint i32 %365, %317
  %409 = or disjoint i32 %408, %281
  %410 = or disjoint i32 %409, %269
  %411 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %410
  %412 = load half, ptr addrspace(3) %411, align 2, !tbaa !41
  %413 = insertelement <4 x half> %407, half %412, i64 1
  %414 = or disjoint i32 %372, %317
  %415 = or disjoint i32 %414, %273
  %416 = or disjoint i32 %415, %269
  %417 = or disjoint i32 %416, 16
  %418 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %417
  %419 = load half, ptr addrspace(3) %418, align 2, !tbaa !41
  %420 = insertelement <4 x half> %413, half %419, i64 2
  %421 = or disjoint i32 %380, %317
  %422 = or disjoint i32 %421, %281
  %423 = or disjoint i32 %422, %269
  %424 = or disjoint i32 %423, 16
  %425 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %424
  %426 = load half, ptr addrspace(3) %425, align 2, !tbaa !41
  %427 = insertelement <4 x half> %420, half %426, i64 3
  %428 = or disjoint i32 %404, 16
  %429 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %428
  %430 = load half, ptr addrspace(3) %429, align 2, !tbaa !41
  %431 = insertelement <4 x half> poison, half %430, i64 0
  %432 = or disjoint i32 %410, 16
  %433 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %432
  %434 = load half, ptr addrspace(3) %433, align 2, !tbaa !41
  %435 = insertelement <4 x half> %431, half %434, i64 1
  %436 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %416
  %437 = load half, ptr addrspace(3) %436, align 2, !tbaa !41
  %438 = insertelement <4 x half> %435, half %437, i64 2
  %439 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %423
  %440 = load half, ptr addrspace(3) %439, align 2, !tbaa !41
  %441 = insertelement <4 x half> %438, half %440, i64 3
  %442 = insertelement <4 x half> %233, half %267, i64 3
  %443 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %302, <4 x half> %442, <4 x float> zeroinitializer)
  %444 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %316, <4 x half> %442, <4 x float> zeroinitializer)
  %445 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %343, <4 x half> %442, <4 x float> zeroinitializer)
  %446 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %357, <4 x half> %442, <4 x float> zeroinitializer)
  %447 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %387, <4 x half> %442, <4 x float> zeroinitializer)
  %448 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %401, <4 x half> %442, <4 x float> zeroinitializer)
  %449 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %427, <4 x half> %442, <4 x float> zeroinitializer)
  %450 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %441, <4 x half> %442, <4 x float> zeroinitializer)
  %451 = and i32 %235, 1920
  %452 = or disjoint i32 %451, %33
  %453 = add nuw nsw i32 %452, %17
  %454 = zext nneg i32 %453 to i64
  %455 = extractelement <4 x float> %443, i64 0
  %456 = extractelement <4 x float> %443, i64 1
  %457 = extractelement <4 x float> %443, i64 2
  %458 = extractelement <4 x float> %443, i64 3
  %459 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %460 = fptrunc float %455 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %459), !noalias !42
  %461 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %462 = fptrunc float %456 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %461), !noalias !42
  %463 = bitcast half %460 to i16
  %464 = bitcast half %462 to i16
  %465 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %466 = fptrunc float %457 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %465), !noalias !47
  %467 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %468 = fptrunc float %458 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %467), !noalias !47
  %469 = bitcast half %466 to i16
  %470 = bitcast half %468 to i16
  %471 = zext i16 %470 to i64
  %472 = shl nuw i64 %471, 48
  %473 = zext i16 %469 to i64
  %474 = shl nuw nsw i64 %473, 32
  %475 = or disjoint i64 %472, %474
  %476 = zext i16 %464 to i64
  %477 = shl nuw nsw i64 %476, 16
  %478 = or disjoint i64 %475, %477
  %479 = zext i16 %463 to i64
  %480 = or disjoint i64 %478, %479
  %481 = getelementptr inbounds %struct.__half, ptr addrspace(1) %2, i64 %454
  store i64 %480, ptr addrspace(1) %481, align 8
  %482 = extractelement <4 x float> %444, i64 0
  %483 = extractelement <4 x float> %444, i64 1
  %484 = extractelement <4 x float> %444, i64 2
  %485 = extractelement <4 x float> %444, i64 3
  %486 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %487 = fptrunc float %482 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %486), !noalias !42
  %488 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %489 = fptrunc float %483 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %488), !noalias !42
  %490 = bitcast half %487 to i16
  %491 = bitcast half %489 to i16
  %492 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %493 = fptrunc float %484 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %492), !noalias !47
  %494 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %495 = fptrunc float %485 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %494), !noalias !47
  %496 = bitcast half %493 to i16
  %497 = bitcast half %495 to i16
  %498 = zext i16 %497 to i64
  %499 = shl nuw i64 %498, 48
  %500 = zext i16 %496 to i64
  %501 = shl nuw nsw i64 %500, 32
  %502 = or disjoint i64 %499, %501
  %503 = zext i16 %491 to i64
  %504 = shl nuw nsw i64 %503, 16
  %505 = or disjoint i64 %502, %504
  %506 = zext i16 %490 to i64
  %507 = or disjoint i64 %505, %506
  %508 = getelementptr inbounds i8, ptr addrspace(1) %481, i64 32
  store i64 %507, ptr addrspace(1) %508, align 8
  %509 = extractelement <4 x float> %445, i64 0
  %510 = extractelement <4 x float> %445, i64 1
  %511 = extractelement <4 x float> %445, i64 2
  %512 = extractelement <4 x float> %445, i64 3
  %513 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %514 = fptrunc float %509 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %513), !noalias !42
  %515 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %516 = fptrunc float %510 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %515), !noalias !42
  %517 = bitcast half %514 to i16
  %518 = bitcast half %516 to i16
  %519 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %520 = fptrunc float %511 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %519), !noalias !47
  %521 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %522 = fptrunc float %512 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %521), !noalias !47
  %523 = bitcast half %520 to i16
  %524 = bitcast half %522 to i16
  %525 = zext i16 %524 to i64
  %526 = shl nuw i64 %525, 48
  %527 = zext i16 %523 to i64
  %528 = shl nuw nsw i64 %527, 32
  %529 = or disjoint i64 %526, %528
  %530 = zext i16 %518 to i64
  %531 = shl nuw nsw i64 %530, 16
  %532 = or disjoint i64 %529, %531
  %533 = zext i16 %517 to i64
  %534 = or disjoint i64 %532, %533
  %535 = getelementptr inbounds i8, ptr addrspace(1) %481, i64 64
  store i64 %534, ptr addrspace(1) %535, align 8
  %536 = extractelement <4 x float> %446, i64 0
  %537 = extractelement <4 x float> %446, i64 1
  %538 = extractelement <4 x float> %446, i64 2
  %539 = extractelement <4 x float> %446, i64 3
  %540 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %541 = fptrunc float %536 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %540), !noalias !42
  %542 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %543 = fptrunc float %537 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %542), !noalias !42
  %544 = bitcast half %541 to i16
  %545 = bitcast half %543 to i16
  %546 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %547 = fptrunc float %538 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %546), !noalias !47
  %548 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %549 = fptrunc float %539 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %548), !noalias !47
  %550 = bitcast half %547 to i16
  %551 = bitcast half %549 to i16
  %552 = zext i16 %551 to i64
  %553 = shl nuw i64 %552, 48
  %554 = zext i16 %550 to i64
  %555 = shl nuw nsw i64 %554, 32
  %556 = or disjoint i64 %553, %555
  %557 = zext i16 %545 to i64
  %558 = shl nuw nsw i64 %557, 16
  %559 = or disjoint i64 %556, %558
  %560 = zext i16 %544 to i64
  %561 = or disjoint i64 %559, %560
  %562 = getelementptr inbounds i8, ptr addrspace(1) %481, i64 96
  store i64 %561, ptr addrspace(1) %562, align 8
  %563 = extractelement <4 x float> %447, i64 0
  %564 = extractelement <4 x float> %447, i64 1
  %565 = extractelement <4 x float> %447, i64 2
  %566 = extractelement <4 x float> %447, i64 3
  %567 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %568 = fptrunc float %563 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %567), !noalias !42
  %569 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %570 = fptrunc float %564 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %569), !noalias !42
  %571 = bitcast half %568 to i16
  %572 = bitcast half %570 to i16
  %573 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %574 = fptrunc float %565 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %573), !noalias !47
  %575 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %576 = fptrunc float %566 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %575), !noalias !47
  %577 = bitcast half %574 to i16
  %578 = bitcast half %576 to i16
  %579 = zext i16 %578 to i64
  %580 = shl nuw i64 %579, 48
  %581 = zext i16 %577 to i64
  %582 = shl nuw nsw i64 %581, 32
  %583 = or disjoint i64 %580, %582
  %584 = zext i16 %572 to i64
  %585 = shl nuw nsw i64 %584, 16
  %586 = or disjoint i64 %583, %585
  %587 = zext i16 %571 to i64
  %588 = or disjoint i64 %586, %587
  %589 = getelementptr inbounds i8, ptr addrspace(1) %481, i64 128
  store i64 %588, ptr addrspace(1) %589, align 8
  %590 = extractelement <4 x float> %448, i64 0
  %591 = extractelement <4 x float> %448, i64 1
  %592 = extractelement <4 x float> %448, i64 2
  %593 = extractelement <4 x float> %448, i64 3
  %594 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %595 = fptrunc float %590 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %594), !noalias !42
  %596 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %597 = fptrunc float %591 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %596), !noalias !42
  %598 = bitcast half %595 to i16
  %599 = bitcast half %597 to i16
  %600 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %601 = fptrunc float %592 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %600), !noalias !47
  %602 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %603 = fptrunc float %593 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %602), !noalias !47
  %604 = bitcast half %601 to i16
  %605 = bitcast half %603 to i16
  %606 = zext i16 %605 to i64
  %607 = shl nuw i64 %606, 48
  %608 = zext i16 %604 to i64
  %609 = shl nuw nsw i64 %608, 32
  %610 = or disjoint i64 %607, %609
  %611 = zext i16 %599 to i64
  %612 = shl nuw nsw i64 %611, 16
  %613 = or disjoint i64 %610, %612
  %614 = zext i16 %598 to i64
  %615 = or disjoint i64 %613, %614
  %616 = getelementptr inbounds i8, ptr addrspace(1) %481, i64 160
  store i64 %615, ptr addrspace(1) %616, align 8
  %617 = extractelement <4 x float> %449, i64 0
  %618 = extractelement <4 x float> %449, i64 1
  %619 = extractelement <4 x float> %449, i64 2
  %620 = extractelement <4 x float> %449, i64 3
  %621 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %622 = fptrunc float %617 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %621), !noalias !42
  %623 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %624 = fptrunc float %618 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %623), !noalias !42
  %625 = bitcast half %622 to i16
  %626 = bitcast half %624 to i16
  %627 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %628 = fptrunc float %619 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %627), !noalias !47
  %629 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %630 = fptrunc float %620 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %629), !noalias !47
  %631 = bitcast half %628 to i16
  %632 = bitcast half %630 to i16
  %633 = zext i16 %632 to i64
  %634 = shl nuw i64 %633, 48
  %635 = zext i16 %631 to i64
  %636 = shl nuw nsw i64 %635, 32
  %637 = or disjoint i64 %634, %636
  %638 = zext i16 %626 to i64
  %639 = shl nuw nsw i64 %638, 16
  %640 = or disjoint i64 %637, %639
  %641 = zext i16 %625 to i64
  %642 = or disjoint i64 %640, %641
  %643 = getelementptr inbounds i8, ptr addrspace(1) %481, i64 192
  store i64 %642, ptr addrspace(1) %643, align 8
  %644 = extractelement <4 x float> %450, i64 0
  %645 = extractelement <4 x float> %450, i64 1
  %646 = extractelement <4 x float> %450, i64 2
  %647 = extractelement <4 x float> %450, i64 3
  %648 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %649 = fptrunc float %644 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %648), !noalias !42
  %650 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %651 = fptrunc float %645 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %650), !noalias !42
  %652 = bitcast half %649 to i16
  %653 = bitcast half %651 to i16
  %654 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %655 = fptrunc float %646 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %654), !noalias !47
  %656 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %657 = fptrunc float %647 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %656), !noalias !47
  %658 = bitcast half %655 to i16
  %659 = bitcast half %657 to i16
  %660 = zext i16 %659 to i64
  %661 = shl nuw i64 %660, 48
  %662 = zext i16 %658 to i64
  %663 = shl nuw nsw i64 %662, 32
  %664 = or disjoint i64 %661, %663
  %665 = zext i16 %653 to i64
  %666 = shl nuw nsw i64 %665, 16
  %667 = or disjoint i64 %664, %666
  %668 = zext i16 %652 to i64
  %669 = or disjoint i64 %667, %668
  %670 = getelementptr inbounds i8, ptr addrspace(1) %481, i64 224
  store i64 %669, ptr addrspace(1) %670, align 8
  br label %671

671:                                              ; preds = %12, %5
  ret void
}

; Function Attrs: nounwind speculatable willreturn memory(none)
declare i32 @llvm.mxc.thread.id.x() #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg) #6

; Function Attrs: convergent nounwind willreturn
declare void @llvm.mxc.barrier.warp() #7

; Function Attrs: convergent nounwind willreturn memory(none)
declare <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>) #8

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #9

; Function Attrs: convergent nounwind willreturn memory(none)
declare i32 @llvm.mxc.mbcnt.lo(i32, i32) #8

; Function Attrs: convergent nounwind willreturn memory(none)
declare i32 @llvm.mxc.mbcnt.hi(i32, i32) #8

; Function Attrs: convergent nounwind willreturn memory(none)
declare i32 @llvm.mxc.bsm.bpermute(i32, i32) #8

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp2.f32(float) #9

; Function Attrs: nounwind speculatable willreturn memory(inaccessiblemem: read)
declare i32 @llvm.mxc.gethwreg(i32 immarg) #10

; Function Attrs: nounwind willreturn
declare void @llvm.mxc.sethwreg(i32 immarg, i32) #4

attributes #0 = { mustprogress noreturn nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" "disable-promote-alloca-to-bsm"="true" "disable-promote-alloca-to-vector"="false" "enable-ldg-bsm-opt"="false" "fixed-function-abi"="true" "metaxgpu-bsm-direct-address"="true" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="512" "metaxgpu-new-streg-abi"="false" "metaxgpu-pk-fma"="false" "metaxgpu-resource-usage"="false" "metaxgpu-sched-select"="default" "metaxgpu-use-dim-intrinsic"="false" "no-trapping-math"="true" "prec-div"="false" "prec-sqrt"="false" "scalarize-global-loads"="true" "shfl-combine"="true" "stack-protector-buffer-size"="8" "target-cpu"="xcore1000" "target-features"="+xcore1000" }
attributes #1 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #2 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" "disable-promote-alloca-to-bsm"="true" "disable-promote-alloca-to-vector"="false" "enable-ldg-bsm-opt"="false" "fixed-function-abi"="true" "metaxgpu-bsm-direct-address"="true" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="512" "metaxgpu-new-streg-abi"="false" "metaxgpu-pk-fma"="false" "metaxgpu-resource-usage"="false" "metaxgpu-sched-select"="default" "metaxgpu-use-dim-intrinsic"="false" "no-trapping-math"="true" "prec-div"="false" "prec-sqrt"="false" "scalarize-global-loads"="true" "shfl-combine"="true" "stack-protector-buffer-size"="8" "target-cpu"="xcore1000" "target-features"="+xcore1000" }
attributes #3 = { nounwind speculatable willreturn memory(none) }
attributes #4 = { nounwind willreturn }
attributes #5 = { convergent mustprogress norecurse nounwind willreturn "denormal-fp-math-f32"="preserve-sign,preserve-sign" "disable-promote-alloca-to-bsm"="true" "disable-promote-alloca-to-vector"="false" "enable-ldg-bsm-opt"="false" "fixed-function-abi"="true" "metaxgpu-bsm-direct-address"="true" "metaxgpu-implicitarg-num-bytes"="80" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="64" "metaxgpu-min-blocks"="1" "metaxgpu-new-streg-abi"="false" "metaxgpu-pk-fma"="false" "metaxgpu-resource-usage"="false" "metaxgpu-sched-select"="default" "metaxgpu-use-dim-intrinsic"="false" "no-trapping-math"="true" "prec-div"="false" "prec-sqrt"="false" "scalarize-global-loads"="true" "shfl-combine"="true" "stack-protector-buffer-size"="8" "target-cpu"="xcore1000" "target-features"="+xcore1000" "uniform-work-group-size"="true" }
attributes #6 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { convergent nounwind willreturn }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #10 = { nounwind speculatable willreturn memory(inaccessiblemem: read) }
attributes #11 = { nomerge }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 1}
!2 = !{i32 1, !"LTOPostLink", i32 1}
!3 = !{!4, !4, i64 0}
!4 = !{!"long", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !10, i64 16}
!8 = !{!"_ZTS21_DynamicParallMqlWrap", !9, i64 0, !9, i64 4, !9, i64 8, !9, i64 12, !10, i64 16, !10, i64 24, !11, i64 32}
!9 = !{!"int", !5, i64 0}
!10 = !{!"any pointer", !5, i64 0}
!11 = !{!"_ZTS28mxc_kernel_dispatch_packet_s", !12, i64 0, !12, i64 2, !12, i64 4, !12, i64 6, !12, i64 8, !12, i64 10, !9, i64 12, !9, i64 16, !9, i64 20, !9, i64 24, !9, i64 28, !4, i64 32, !10, i64 40, !4, i64 48, !13, i64 56}
!12 = !{!"short", !5, i64 0}
!13 = !{!"_ZTS12mxc_signal_s", !4, i64 0}
!14 = !{i32 1, i32 -2147483648}
!15 = !{}
!16 = !{i32 0, i32 2147483647}
!17 = !{!9, !9, i64 0}
!18 = distinct !{!18, !19}
!19 = !{!"llvm.loop.mustprogress"}
!20 = !{!8, !9, i64 4}
!21 = !{!8, !10, i64 24}
!22 = !{!8, !9, i64 0}
!23 = !{!24, !10, i64 24}
!24 = !{!"_ZTS28_DynamicParallSchedulerParam", !4, i64 0, !10, i64 8, !10, i64 16, !10, i64 24, !4, i64 32, !4, i64 40, !4, i64 48, !4, i64 56, !4, i64 64, !4, i64 72, !4, i64 80, !4, i64 88, !4, i64 96, !9, i64 104, !9, i64 108, !9, i64 112, !9, i64 116, !9, i64 120, !9, i64 124}
!25 = !{!24, !4, i64 80}
!26 = !{i32 0, i32 1024}
!27 = !{i64 0, i64 4, !17, i64 4, i64 4, !17, i64 8, i64 4, !17, i64 12, i64 4, !17}
!28 = !{i32 -1, i32 3, i32 -1, i32 -1}
!29 = !{i32 -1, i32 1, i32 -1, i32 -1}
!30 = !{!31, !33}
!31 = distinct !{!31, !32, !"_ZL17__floats2half2_rnff: %agg.result"}
!32 = distinct !{!32, !"_ZL17__floats2half2_rnff"}
!33 = distinct !{!33, !34, !"_ZL17__float22half2_rn6float2: %agg.result"}
!34 = distinct !{!34, !"_ZL17__float22half2_rn6float2"}
!35 = !{!36, !38}
!36 = distinct !{!36, !37, !"_ZL17__floats2half2_rnff: %agg.result"}
!37 = distinct !{!37, !"_ZL17__floats2half2_rnff"}
!38 = distinct !{!38, !39, !"_ZL17__float22half2_rn6float2: %agg.result"}
!39 = distinct !{!39, !"_ZL17__float22half2_rn6float2"}
!40 = !{i32 -1, i32 4, i32 -1, i32 -1}
!41 = !{!12, !12, i64 0}
!42 = !{!43, !45}
!43 = distinct !{!43, !44, !"_ZL17__floats2half2_rnff: %agg.result"}
!44 = distinct !{!44, !"_ZL17__floats2half2_rnff"}
!45 = distinct !{!45, !46, !"_ZL17__float22half2_rn6float2: %agg.result"}
!46 = distinct !{!46, !"_ZL17__float22half2_rn6float2"}
!47 = !{!48, !50}
!48 = distinct !{!48, !49, !"_ZL17__floats2half2_rnff: %agg.result"}
!49 = distinct !{!49, !"_ZL17__floats2half2_rnff"}
!50 = distinct !{!50, !51, !"_ZL17__float22half2_rn6float2: %agg.result"}
!51 = distinct !{!51, !"_ZL17__float22half2_rn6float2"}
