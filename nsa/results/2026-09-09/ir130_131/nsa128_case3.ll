; ModuleID = '/data/nsa_codex_20260909_r1/results/ir130_131/nsa128_case3.bc'
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
  %10 = shl nsw i32 %9, 4
  %11 = icmp sgt i32 %10, %6
  br i1 %11, label %801, label %12

12:                                               ; preds = %5
  %13 = tail call noundef range(i32 0, 1024) i32 @llvm.mxc.thread.id.x(), !range !26
  %14 = lshr i32 %13, 2
  %15 = and i32 %14, 252
  %16 = add nsw i32 %10, %15
  %17 = icmp sgt i32 %16, %6
  %18 = select i1 %17, float 0xFFF0000000000000, float 0.000000e+00
  %19 = insertelement <4 x float> poison, float %18, i64 0
  %20 = icmp slt i32 %16, %6
  %21 = select i1 %20, float 0.000000e+00, float 0xFFF0000000000000
  %22 = insertelement <4 x float> %19, float %21, i64 1
  %23 = or disjoint i32 %16, 2
  %24 = icmp sgt i32 %23, %6
  %25 = select i1 %24, float 0xFFF0000000000000, float 0.000000e+00
  %26 = insertelement <4 x float> %22, float %25, i64 2
  %27 = or disjoint i32 %16, 3
  %28 = icmp sgt i32 %27, %6
  %29 = select i1 %28, float 0xFFF0000000000000, float 0.000000e+00
  %30 = insertelement <4 x float> %26, float %29, i64 3
  %31 = shl nuw nsw i64 %7, 11
  %32 = shl nuw nsw i32 %13, 4
  %33 = and i32 %32, 16256
  %34 = zext nneg i32 %33 to i64
  %35 = add nuw nsw i64 %31, %34
  %36 = shl nuw nsw i32 %13, 3
  %37 = and i32 %36, 56
  %38 = zext nneg i32 %37 to i64
  %39 = or disjoint i64 %35, %38
  %40 = and i32 %36, 8128
  %41 = and i32 %36, 32
  %42 = add nuw nsw i32 %41, %13
  %43 = and i32 %42, 32
  %44 = and i32 %36, 16
  %45 = add nuw nsw i32 %44, %13
  %46 = and i32 %45, 16
  %47 = mul nuw nsw i32 %13, 9
  %48 = and i32 %47, 8
  %49 = sext i32 %10 to i64
  %50 = shl nsw i64 %49, 7
  %51 = shl nuw nsw i32 %13, 6
  %52 = and i32 %51, 960
  %53 = lshr i32 %13, 1
  %54 = lshr i32 %13, 5
  %55 = add nuw nsw i32 %54, %13
  %56 = shl nuw nsw i32 %55, 3
  %57 = and i32 %56, 8
  %58 = and i32 %14, 4
  %59 = or disjoint i32 %58, %52
  %60 = or disjoint i32 %59, %57
  %61 = getelementptr inbounds %struct.__half, ptr addrspace(4) %3, i64 %39
  %62 = or disjoint i32 %40, %43
  %63 = or disjoint i32 %62, %46
  %64 = or disjoint i32 %63, %48
  %65 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 2048), i32 %64
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %65, ptr addrspace(4) noundef align 16 dereferenceable(16) %61, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %66 = getelementptr inbounds i8, ptr addrspace(4) %61, i64 2048
  %67 = add nuw nsw i32 %40, 512
  %68 = or disjoint i32 %67, %43
  %69 = or disjoint i32 %68, %46
  %70 = or disjoint i32 %69, %48
  %71 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 2048), i32 %70
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %71, ptr addrspace(4) noundef align 16 dereferenceable(16) %66, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %72 = add nsw i64 %50, %34
  %73 = or disjoint i64 %72, %38
  %74 = getelementptr inbounds %struct.__half, ptr addrspace(4) %1, i64 %73
  %75 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %64
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %75, ptr addrspace(4) noundef align 16 dereferenceable(16) %74, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  %76 = add nuw nsw i64 %34, 1024
  %77 = add nsw i64 %50, %76
  %78 = or disjoint i64 %77, %38
  %79 = getelementptr inbounds %struct.__half, ptr addrspace(4) %1, i64 %78
  %80 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %70
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %80, ptr addrspace(4) noundef align 16 dereferenceable(16) %79, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  %81 = shl nuw nsw i32 %14, 5
  %82 = and i32 %81, 32
  %83 = shl nuw nsw i32 %53, 4
  %84 = and i32 %83, 16
  %85 = or disjoint i32 %60, %84
  %86 = or disjoint i32 %85, %82
  %87 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 2048), i32 %86
  %88 = load <4 x half>, ptr addrspace(3) %87, align 8
  %89 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %86
  %90 = load <4 x half>, ptr addrspace(3) %89, align 8
  %91 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %90, <4 x half> %88, <4 x float> %30)
  %92 = xor i32 %86, 16
  %93 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 2048), i32 %92
  %94 = load <4 x half>, ptr addrspace(3) %93, align 8
  %95 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %92
  %96 = load <4 x half>, ptr addrspace(3) %95, align 8
  %97 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %96, <4 x half> %94, <4 x float> %91)
  %98 = or disjoint i32 %82, %85
  %99 = xor i32 %98, 32
  %100 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 2048), i32 %99
  %101 = load <4 x half>, ptr addrspace(3) %100, align 8
  %102 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %99
  %103 = load <4 x half>, ptr addrspace(3) %102, align 8
  %104 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %103, <4 x half> %101, <4 x float> %97)
  %105 = xor i32 %98, 48
  %106 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 2048), i32 %105
  %107 = load <4 x half>, ptr addrspace(3) %106, align 8
  %108 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %105
  %109 = load <4 x half>, ptr addrspace(3) %108, align 8
  %110 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %109, <4 x half> %107, <4 x float> %104)
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  %111 = getelementptr inbounds i8, ptr addrspace(4) %61, i64 128
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %65, ptr addrspace(4) noundef align 16 dereferenceable(16) %111, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %112 = getelementptr inbounds i8, ptr addrspace(4) %61, i64 2176
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %71, ptr addrspace(4) noundef align 16 dereferenceable(16) %112, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %113 = or disjoint i64 %73, 64
  %114 = getelementptr inbounds %struct.__half, ptr addrspace(4) %1, i64 %113
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %75, ptr addrspace(4) noundef align 16 dereferenceable(16) %114, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  %115 = or disjoint i64 %78, 64
  %116 = getelementptr inbounds %struct.__half, ptr addrspace(4) %1, i64 %115
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %80, ptr addrspace(4) noundef align 16 dereferenceable(16) %116, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  %117 = load <4 x half>, ptr addrspace(3) %87, align 8
  %118 = load <4 x half>, ptr addrspace(3) %89, align 8
  %119 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %118, <4 x half> %117, <4 x float> %110)
  %120 = load <4 x half>, ptr addrspace(3) %93, align 8
  %121 = load <4 x half>, ptr addrspace(3) %95, align 8
  %122 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %121, <4 x half> %120, <4 x float> %119)
  %123 = load <4 x half>, ptr addrspace(3) %100, align 8
  %124 = load <4 x half>, ptr addrspace(3) %102, align 8
  %125 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %124, <4 x half> %123, <4 x float> %122)
  %126 = load <4 x half>, ptr addrspace(3) %106, align 8
  %127 = load <4 x half>, ptr addrspace(3) %108, align 8
  %128 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %127, <4 x half> %126, <4 x float> %125)
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  %129 = extractelement <4 x float> %128, i64 0
  %130 = tail call contract noundef float @llvm.maxnum.f32(float %129, float 0xFFF0000000000000)
  %131 = extractelement <4 x float> %128, i64 1
  %132 = tail call contract noundef float @llvm.maxnum.f32(float %130, float %131)
  %133 = extractelement <4 x float> %128, i64 2
  %134 = tail call contract noundef float @llvm.maxnum.f32(float %132, float %133)
  %135 = extractelement <4 x float> %128, i64 3
  %136 = tail call contract noundef float @llvm.maxnum.f32(float %134, float %135)
  %137 = bitcast float %136 to i32
  %138 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #11
  %139 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %138) #11
  %140 = xor i32 %139, 32
  %141 = and i32 %139, -64
  %142 = add nsw i32 %141, 64
  %143 = icmp slt i32 %140, %142
  %144 = select i1 %143, i32 %140, i32 %139
  %145 = shl i32 %144, 2
  %146 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %145, i32 %137)
  %147 = bitcast i32 %146 to float
  %148 = tail call contract noundef float @llvm.maxnum.f32(float %136, float %147)
  %149 = bitcast float %148 to i32
  %150 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #11
  %151 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %150) #11
  %152 = xor i32 %151, 16
  %153 = and i32 %151, -64
  %154 = add nsw i32 %153, 64
  %155 = icmp slt i32 %152, %154
  %156 = select i1 %155, i32 %152, i32 %151
  %157 = shl i32 %156, 2
  %158 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %157, i32 %149)
  %159 = bitcast i32 %158 to float
  %160 = tail call contract noundef float @llvm.maxnum.f32(float %148, float %159)
  %161 = fsub contract float %129, %160
  %162 = fmul contract float %161, 0x3FC0527DC0000000
  %163 = fcmp contract olt float %162, -1.260000e+02
  %164 = select contract i1 %163, float 6.400000e+01, float 0.000000e+00
  %165 = fadd contract float %162, %164
  %166 = tail call contract float @llvm.exp2.f32(float %165)
  %167 = select contract i1 %163, float 0x3BF0000000000000, float 1.000000e+00
  %168 = fmul contract float %167, %166
  %169 = fsub contract float %131, %160
  %170 = fmul contract float %169, 0x3FC0527DC0000000
  %171 = fcmp contract olt float %170, -1.260000e+02
  %172 = select contract i1 %171, float 6.400000e+01, float 0.000000e+00
  %173 = fadd contract float %170, %172
  %174 = tail call contract float @llvm.exp2.f32(float %173)
  %175 = select contract i1 %171, float 0x3BF0000000000000, float 1.000000e+00
  %176 = fmul contract float %175, %174
  %177 = fsub contract float %133, %160
  %178 = fmul contract float %177, 0x3FC0527DC0000000
  %179 = fcmp contract olt float %178, -1.260000e+02
  %180 = select contract i1 %179, float 6.400000e+01, float 0.000000e+00
  %181 = fadd contract float %178, %180
  %182 = tail call contract float @llvm.exp2.f32(float %181)
  %183 = select contract i1 %179, float 0x3BF0000000000000, float 1.000000e+00
  %184 = fmul contract float %183, %182
  %185 = fsub contract float %135, %160
  %186 = fmul contract float %185, 0x3FC0527DC0000000
  %187 = fcmp contract olt float %186, -1.260000e+02
  %188 = select contract i1 %187, float 6.400000e+01, float 0.000000e+00
  %189 = fadd contract float %186, %188
  %190 = tail call contract float @llvm.exp2.f32(float %189)
  %191 = select contract i1 %187, float 0x3BF0000000000000, float 1.000000e+00
  %192 = fmul contract float %191, %190
  %193 = fadd contract float %168, 0.000000e+00
  %194 = fadd contract float %193, %176
  %195 = fadd contract float %194, %184
  %196 = fadd contract float %195, %192
  %197 = bitcast float %196 to i32
  %198 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #11
  %199 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %198) #11
  %200 = xor i32 %199, 32
  %201 = and i32 %199, -64
  %202 = add nsw i32 %201, 64
  %203 = icmp slt i32 %200, %202
  %204 = select i1 %203, i32 %200, i32 %199
  %205 = shl i32 %204, 2
  %206 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %205, i32 %197)
  %207 = bitcast i32 %206 to float
  %208 = fadd contract float %196, %207
  %209 = bitcast float %208 to i32
  %210 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #11
  %211 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %210) #11
  %212 = xor i32 %211, 16
  %213 = and i32 %211, -64
  %214 = add nsw i32 %213, 64
  %215 = icmp slt i32 %212, %214
  %216 = select i1 %215, i32 %212, i32 %211
  %217 = shl i32 %216, 2
  %218 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %217, i32 %209)
  %219 = bitcast i32 %218 to float
  %220 = fadd contract float %208, %219
  %221 = fdiv contract float %168, %220
  %222 = fdiv contract float %176, %220
  %223 = fdiv contract float %184, %220
  %224 = fdiv contract float %192, %220
  %225 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !30
  %226 = fptrunc float %221 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %225), !noalias !30
  %227 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !30
  %228 = fptrunc float %222 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %227), !noalias !30
  %229 = insertelement <4 x half> poison, half %226, i64 0
  %230 = insertelement <4 x half> %229, half %228, i64 1
  %231 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !35
  %232 = fptrunc float %223 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %231), !noalias !35
  %233 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !35
  %234 = fptrunc float %224 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %233), !noalias !35
  %235 = insertelement <4 x half> %230, half %232, i64 2
  %236 = insertelement <4 x half> %235, half %234, i64 3
  %237 = icmp ult i32 %9, 16
  br i1 %237, label %260, label %238

238:                                              ; preds = %12
  %239 = lshr i32 %13, 4
  %240 = add i32 %10, %239
  %241 = ashr i32 %240, 2
  %242 = shl nuw nsw i32 %13, 7
  %243 = and i32 %242, 1024
  %244 = shl nuw nsw i32 %13, 2
  %245 = and i32 %244, 4032
  %246 = add nuw nsw i32 %245, %243
  %247 = add nuw nsw i32 %54, %53
  %248 = shl nuw nsw i32 %247, 4
  %249 = and i32 %248, 16
  %250 = add nuw nsw i32 %239, %13
  %251 = shl nuw nsw i32 %250, 3
  %252 = and i32 %251, 8
  %253 = or disjoint i32 %246, %249
  %254 = or disjoint i32 %253, %252
  %255 = zext nneg i32 %36 to i64
  %256 = getelementptr %struct.__half, ptr addrspace(4) %4, i64 %255
  %257 = icmp slt i32 %241, 64
  %258 = icmp sgt i32 %240, -1
  %259 = and i1 %257, %258
  br i1 %259, label %295, label %305

260:                                              ; preds = %12
  %261 = shl nuw nsw i32 %9, 11
  %262 = add nuw nsw i32 %261, %36
  %263 = shl nuw nsw i32 %13, 7
  %264 = and i32 %263, 1024
  %265 = shl nuw nsw i32 %13, 2
  %266 = and i32 %265, 4032
  %267 = add nuw nsw i32 %266, %264
  %268 = add nuw nsw i32 %54, %53
  %269 = shl nuw nsw i32 %268, 4
  %270 = and i32 %269, 16
  %271 = lshr i32 %13, 4
  %272 = add nuw nsw i32 %271, %13
  %273 = shl nuw nsw i32 %272, 3
  %274 = and i32 %273, 8
  %275 = or disjoint i32 %267, %270
  %276 = or disjoint i32 %275, %274
  %277 = zext nneg i32 %262 to i64
  %278 = getelementptr inbounds %struct.__half, ptr addrspace(4) %4, i64 %277
  %279 = or disjoint i32 %276, %82
  %280 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %279
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %280, ptr addrspace(4) noundef align 16 dereferenceable(16) %278, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !40
  %281 = getelementptr inbounds i8, ptr addrspace(4) %278, i64 1024
  %282 = add nuw nsw i32 %276, 256
  %283 = or disjoint i32 %82, %282
  %284 = xor i32 %283, 32
  %285 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %284
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %285, ptr addrspace(4) noundef align 16 dereferenceable(16) %281, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !40
  %286 = getelementptr inbounds i8, ptr addrspace(4) %278, i64 2048
  %287 = add nuw nsw i32 %276, 512
  %288 = or disjoint i32 %287, %82
  %289 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %288
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %289, ptr addrspace(4) noundef align 16 dereferenceable(16) %286, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !40
  %290 = getelementptr inbounds i8, ptr addrspace(4) %278, i64 3072
  %291 = add nuw nsw i32 %276, 768
  %292 = or disjoint i32 %82, %291
  %293 = xor i32 %292, 32
  %294 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %293
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %294, ptr addrspace(4) noundef align 16 dereferenceable(16) %290, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !40
  br label %395

295:                                              ; preds = %238
  %296 = shl nsw i64 %49, 8
  %297 = getelementptr i8, ptr addrspace(4) %256, i64 %296
  %298 = load i32, ptr addrspace(4) %297, align 16, !tbaa !17
  %299 = getelementptr inbounds i8, ptr addrspace(4) %297, i64 4
  %300 = load i32, ptr addrspace(4) %299, align 4, !tbaa !17
  %301 = getelementptr inbounds i8, ptr addrspace(4) %297, i64 8
  %302 = load i32, ptr addrspace(4) %301, align 8, !tbaa !17
  %303 = getelementptr inbounds i8, ptr addrspace(4) %297, i64 12
  %304 = load i32, ptr addrspace(4) %303, align 4, !tbaa !17
  br label %305

305:                                              ; preds = %295, %238
  %306 = phi i32 [ %298, %295 ], [ 0, %238 ]
  %307 = phi i32 [ %300, %295 ], [ 0, %238 ]
  %308 = phi i32 [ %302, %295 ], [ 0, %238 ]
  %309 = phi i32 [ %304, %295 ], [ 0, %238 ]
  %310 = or disjoint i32 %254, %82
  %311 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %310
  store i32 %306, ptr addrspace(3) %311, align 16, !tbaa !17
  %312 = getelementptr inbounds i8, ptr addrspace(3) %311, i32 4
  store i32 %307, ptr addrspace(3) %312, align 4, !tbaa !17
  %313 = getelementptr inbounds i8, ptr addrspace(3) %311, i32 8
  store i32 %308, ptr addrspace(3) %313, align 8, !tbaa !17
  %314 = getelementptr inbounds i8, ptr addrspace(3) %311, i32 12
  store i32 %309, ptr addrspace(3) %314, align 4, !tbaa !17
  %315 = icmp slt i32 %241, 63
  %316 = add i32 %240, 4
  %317 = icmp sgt i32 %316, -1
  %318 = and i1 %315, %317
  br i1 %318, label %319, label %330

319:                                              ; preds = %305
  %320 = shl nsw i64 %49, 8
  %321 = getelementptr i8, ptr addrspace(4) %256, i64 %320
  %322 = getelementptr i8, ptr addrspace(4) %321, i64 1024
  %323 = load i32, ptr addrspace(4) %322, align 16, !tbaa !17
  %324 = getelementptr i8, ptr addrspace(4) %321, i64 1028
  %325 = load i32, ptr addrspace(4) %324, align 4, !tbaa !17
  %326 = getelementptr i8, ptr addrspace(4) %321, i64 1032
  %327 = load i32, ptr addrspace(4) %326, align 8, !tbaa !17
  %328 = getelementptr i8, ptr addrspace(4) %321, i64 1036
  %329 = load i32, ptr addrspace(4) %328, align 4, !tbaa !17
  br label %330

330:                                              ; preds = %319, %305
  %331 = phi i32 [ %323, %319 ], [ 0, %305 ]
  %332 = phi i32 [ %325, %319 ], [ 0, %305 ]
  %333 = phi i32 [ %327, %319 ], [ 0, %305 ]
  %334 = phi i32 [ %329, %319 ], [ 0, %305 ]
  %335 = add nuw nsw i32 %254, 256
  %336 = or disjoint i32 %82, %335
  %337 = xor i32 %336, 32
  %338 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %337
  store i32 %331, ptr addrspace(3) %338, align 16, !tbaa !17
  %339 = getelementptr inbounds i8, ptr addrspace(3) %338, i32 4
  store i32 %332, ptr addrspace(3) %339, align 4, !tbaa !17
  %340 = getelementptr inbounds i8, ptr addrspace(3) %338, i32 8
  store i32 %333, ptr addrspace(3) %340, align 8, !tbaa !17
  %341 = getelementptr inbounds i8, ptr addrspace(3) %338, i32 12
  store i32 %334, ptr addrspace(3) %341, align 4, !tbaa !17
  %342 = icmp slt i32 %241, 62
  %343 = add i32 %240, 8
  %344 = icmp sgt i32 %343, -1
  %345 = and i1 %342, %344
  br i1 %345, label %346, label %357

346:                                              ; preds = %330
  %347 = shl nsw i64 %49, 8
  %348 = getelementptr i8, ptr addrspace(4) %256, i64 %347
  %349 = getelementptr i8, ptr addrspace(4) %348, i64 2048
  %350 = load i32, ptr addrspace(4) %349, align 16, !tbaa !17
  %351 = getelementptr i8, ptr addrspace(4) %348, i64 2052
  %352 = load i32, ptr addrspace(4) %351, align 4, !tbaa !17
  %353 = getelementptr i8, ptr addrspace(4) %348, i64 2056
  %354 = load i32, ptr addrspace(4) %353, align 8, !tbaa !17
  %355 = getelementptr i8, ptr addrspace(4) %348, i64 2060
  %356 = load i32, ptr addrspace(4) %355, align 4, !tbaa !17
  br label %357

357:                                              ; preds = %346, %330
  %358 = phi i32 [ %350, %346 ], [ 0, %330 ]
  %359 = phi i32 [ %352, %346 ], [ 0, %330 ]
  %360 = phi i32 [ %354, %346 ], [ 0, %330 ]
  %361 = phi i32 [ %356, %346 ], [ 0, %330 ]
  %362 = add nuw nsw i32 %254, 512
  %363 = or disjoint i32 %362, %82
  %364 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %363
  store i32 %358, ptr addrspace(3) %364, align 16, !tbaa !17
  %365 = getelementptr inbounds i8, ptr addrspace(3) %364, i32 4
  store i32 %359, ptr addrspace(3) %365, align 4, !tbaa !17
  %366 = getelementptr inbounds i8, ptr addrspace(3) %364, i32 8
  store i32 %360, ptr addrspace(3) %366, align 8, !tbaa !17
  %367 = getelementptr inbounds i8, ptr addrspace(3) %364, i32 12
  store i32 %361, ptr addrspace(3) %367, align 4, !tbaa !17
  %368 = icmp slt i32 %241, 61
  %369 = add i32 %240, 12
  %370 = icmp sgt i32 %369, -1
  %371 = and i1 %368, %370
  br i1 %371, label %372, label %383

372:                                              ; preds = %357
  %373 = shl nsw i64 %49, 8
  %374 = getelementptr i8, ptr addrspace(4) %256, i64 %373
  %375 = getelementptr i8, ptr addrspace(4) %374, i64 3072
  %376 = load i32, ptr addrspace(4) %375, align 16, !tbaa !17
  %377 = getelementptr i8, ptr addrspace(4) %374, i64 3076
  %378 = load i32, ptr addrspace(4) %377, align 4, !tbaa !17
  %379 = getelementptr i8, ptr addrspace(4) %374, i64 3080
  %380 = load i32, ptr addrspace(4) %379, align 8, !tbaa !17
  %381 = getelementptr i8, ptr addrspace(4) %374, i64 3084
  %382 = load i32, ptr addrspace(4) %381, align 4, !tbaa !17
  br label %383

383:                                              ; preds = %372, %357
  %384 = phi i32 [ %376, %372 ], [ 0, %357 ]
  %385 = phi i32 [ %378, %372 ], [ 0, %357 ]
  %386 = phi i32 [ %380, %372 ], [ 0, %357 ]
  %387 = phi i32 [ %382, %372 ], [ 0, %357 ]
  %388 = add nuw nsw i32 %254, 768
  %389 = or disjoint i32 %82, %388
  %390 = xor i32 %389, 32
  %391 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %390
  store i32 %384, ptr addrspace(3) %391, align 16, !tbaa !17
  %392 = getelementptr inbounds i8, ptr addrspace(3) %391, i32 4
  store i32 %385, ptr addrspace(3) %392, align 4, !tbaa !17
  %393 = getelementptr inbounds i8, ptr addrspace(3) %391, i32 8
  store i32 %386, ptr addrspace(3) %393, align 8, !tbaa !17
  %394 = getelementptr inbounds i8, ptr addrspace(3) %391, i32 12
  store i32 %387, ptr addrspace(3) %394, align 4, !tbaa !17
  br label %395

395:                                              ; preds = %383, %260
  %396 = phi i32 [ %242, %383 ], [ %263, %260 ]
  %397 = phi i32 [ %239, %383 ], [ %271, %260 ]
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  %398 = and i32 %32, 16128
  %399 = and i32 %13, 7
  %400 = shl nuw nsw i32 %397, 5
  %401 = and i32 %400, 32
  %402 = or disjoint i32 %398, %401
  %403 = and i32 %13, 8
  %404 = or disjoint i32 %402, %403
  %405 = or disjoint i32 %404, %399
  %406 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %405
  %407 = load half, ptr addrspace(3) %406, align 2, !tbaa !41
  %408 = insertelement <4 x half> poison, half %407, i64 0
  %409 = or disjoint i32 %398, 64
  %410 = or disjoint i32 %409, %401
  %411 = xor i32 %403, 8
  %412 = or disjoint i32 %410, %411
  %413 = or disjoint i32 %412, %399
  %414 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %413
  %415 = load half, ptr addrspace(3) %414, align 2, !tbaa !41
  %416 = insertelement <4 x half> %408, half %415, i64 1
  %417 = or disjoint i32 %398, 128
  %418 = or disjoint i32 %417, %401
  %419 = or disjoint i32 %418, %403
  %420 = or disjoint i32 %419, %399
  %421 = or disjoint i32 %420, 16
  %422 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %421
  %423 = load half, ptr addrspace(3) %422, align 2, !tbaa !41
  %424 = insertelement <4 x half> %416, half %423, i64 2
  %425 = or disjoint i32 %398, 192
  %426 = or disjoint i32 %425, %401
  %427 = or disjoint i32 %426, %411
  %428 = or disjoint i32 %427, %399
  %429 = or disjoint i32 %428, 16
  %430 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %429
  %431 = load half, ptr addrspace(3) %430, align 2, !tbaa !41
  %432 = insertelement <4 x half> %424, half %431, i64 3
  %433 = or disjoint i32 %405, 16
  %434 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %433
  %435 = load half, ptr addrspace(3) %434, align 2, !tbaa !41
  %436 = insertelement <4 x half> poison, half %435, i64 0
  %437 = or disjoint i32 %413, 16
  %438 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %437
  %439 = load half, ptr addrspace(3) %438, align 2, !tbaa !41
  %440 = insertelement <4 x half> %436, half %439, i64 1
  %441 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %420
  %442 = load half, ptr addrspace(3) %441, align 2, !tbaa !41
  %443 = insertelement <4 x half> %440, half %442, i64 2
  %444 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %428
  %445 = load half, ptr addrspace(3) %444, align 2, !tbaa !41
  %446 = insertelement <4 x half> %443, half %445, i64 3
  %447 = xor i32 %401, 32
  %448 = or disjoint i32 %398, %447
  %449 = or disjoint i32 %448, %403
  %450 = or disjoint i32 %449, %399
  %451 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %450
  %452 = load half, ptr addrspace(3) %451, align 2, !tbaa !41
  %453 = insertelement <4 x half> poison, half %452, i64 0
  %454 = or disjoint i32 %409, %447
  %455 = or disjoint i32 %454, %411
  %456 = or disjoint i32 %455, %399
  %457 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %456
  %458 = load half, ptr addrspace(3) %457, align 2, !tbaa !41
  %459 = insertelement <4 x half> %453, half %458, i64 1
  %460 = or disjoint i32 %417, %447
  %461 = or disjoint i32 %460, %403
  %462 = or disjoint i32 %461, %399
  %463 = or disjoint i32 %462, 16
  %464 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %463
  %465 = load half, ptr addrspace(3) %464, align 2, !tbaa !41
  %466 = insertelement <4 x half> %459, half %465, i64 2
  %467 = or disjoint i32 %425, %447
  %468 = or disjoint i32 %467, %411
  %469 = or disjoint i32 %468, %399
  %470 = or disjoint i32 %469, 16
  %471 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %470
  %472 = load half, ptr addrspace(3) %471, align 2, !tbaa !41
  %473 = insertelement <4 x half> %466, half %472, i64 3
  %474 = or disjoint i32 %450, 16
  %475 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %474
  %476 = load half, ptr addrspace(3) %475, align 2, !tbaa !41
  %477 = insertelement <4 x half> poison, half %476, i64 0
  %478 = or disjoint i32 %456, 16
  %479 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %478
  %480 = load half, ptr addrspace(3) %479, align 2, !tbaa !41
  %481 = insertelement <4 x half> %477, half %480, i64 1
  %482 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %462
  %483 = load half, ptr addrspace(3) %482, align 2, !tbaa !41
  %484 = insertelement <4 x half> %481, half %483, i64 2
  %485 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %469
  %486 = load half, ptr addrspace(3) %485, align 2, !tbaa !41
  %487 = insertelement <4 x half> %484, half %486, i64 3
  %488 = add nuw nsw i32 %398, 1024
  %489 = or disjoint i32 %488, %401
  %490 = or disjoint i32 %489, %403
  %491 = or disjoint i32 %490, %399
  %492 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %491
  %493 = load half, ptr addrspace(3) %492, align 2, !tbaa !41
  %494 = insertelement <4 x half> poison, half %493, i64 0
  %495 = add nuw nsw i32 %398, 1088
  %496 = or disjoint i32 %495, %401
  %497 = or disjoint i32 %496, %411
  %498 = or disjoint i32 %497, %399
  %499 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %498
  %500 = load half, ptr addrspace(3) %499, align 2, !tbaa !41
  %501 = insertelement <4 x half> %494, half %500, i64 1
  %502 = add nuw nsw i32 %398, 1152
  %503 = or disjoint i32 %502, %401
  %504 = or disjoint i32 %503, %403
  %505 = or disjoint i32 %504, %399
  %506 = or disjoint i32 %505, 16
  %507 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %506
  %508 = load half, ptr addrspace(3) %507, align 2, !tbaa !41
  %509 = insertelement <4 x half> %501, half %508, i64 2
  %510 = add nuw nsw i32 %398, 1216
  %511 = or disjoint i32 %510, %401
  %512 = or disjoint i32 %511, %411
  %513 = or disjoint i32 %512, %399
  %514 = or disjoint i32 %513, 16
  %515 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %514
  %516 = load half, ptr addrspace(3) %515, align 2, !tbaa !41
  %517 = insertelement <4 x half> %509, half %516, i64 3
  %518 = or disjoint i32 %491, 16
  %519 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %518
  %520 = load half, ptr addrspace(3) %519, align 2, !tbaa !41
  %521 = insertelement <4 x half> poison, half %520, i64 0
  %522 = or disjoint i32 %498, 16
  %523 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %522
  %524 = load half, ptr addrspace(3) %523, align 2, !tbaa !41
  %525 = insertelement <4 x half> %521, half %524, i64 1
  %526 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %505
  %527 = load half, ptr addrspace(3) %526, align 2, !tbaa !41
  %528 = insertelement <4 x half> %525, half %527, i64 2
  %529 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %513
  %530 = load half, ptr addrspace(3) %529, align 2, !tbaa !41
  %531 = insertelement <4 x half> %528, half %530, i64 3
  %532 = or disjoint i32 %488, %447
  %533 = or disjoint i32 %532, %403
  %534 = or disjoint i32 %533, %399
  %535 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %534
  %536 = load half, ptr addrspace(3) %535, align 2, !tbaa !41
  %537 = insertelement <4 x half> poison, half %536, i64 0
  %538 = or disjoint i32 %495, %447
  %539 = or disjoint i32 %538, %411
  %540 = or disjoint i32 %539, %399
  %541 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %540
  %542 = load half, ptr addrspace(3) %541, align 2, !tbaa !41
  %543 = insertelement <4 x half> %537, half %542, i64 1
  %544 = or disjoint i32 %502, %447
  %545 = or disjoint i32 %544, %403
  %546 = or disjoint i32 %545, %399
  %547 = or disjoint i32 %546, 16
  %548 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %547
  %549 = load half, ptr addrspace(3) %548, align 2, !tbaa !41
  %550 = insertelement <4 x half> %543, half %549, i64 2
  %551 = or disjoint i32 %510, %447
  %552 = or disjoint i32 %551, %411
  %553 = or disjoint i32 %552, %399
  %554 = or disjoint i32 %553, 16
  %555 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %554
  %556 = load half, ptr addrspace(3) %555, align 2, !tbaa !41
  %557 = insertelement <4 x half> %550, half %556, i64 3
  %558 = or disjoint i32 %534, 16
  %559 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %558
  %560 = load half, ptr addrspace(3) %559, align 2, !tbaa !41
  %561 = insertelement <4 x half> poison, half %560, i64 0
  %562 = or disjoint i32 %540, 16
  %563 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %562
  %564 = load half, ptr addrspace(3) %563, align 2, !tbaa !41
  %565 = insertelement <4 x half> %561, half %564, i64 1
  %566 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %546
  %567 = load half, ptr addrspace(3) %566, align 2, !tbaa !41
  %568 = insertelement <4 x half> %565, half %567, i64 2
  %569 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %553
  %570 = load half, ptr addrspace(3) %569, align 2, !tbaa !41
  %571 = insertelement <4 x half> %568, half %570, i64 3
  %572 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %432, <4 x half> %236, <4 x float> zeroinitializer)
  %573 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %446, <4 x half> %236, <4 x float> zeroinitializer)
  %574 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %473, <4 x half> %236, <4 x float> zeroinitializer)
  %575 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %487, <4 x half> %236, <4 x float> zeroinitializer)
  %576 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %517, <4 x half> %236, <4 x float> zeroinitializer)
  %577 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %531, <4 x half> %236, <4 x float> zeroinitializer)
  %578 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %557, <4 x half> %236, <4 x float> zeroinitializer)
  %579 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %571, <4 x half> %236, <4 x float> zeroinitializer)
  %580 = and i32 %396, 1920
  %581 = zext nneg i32 %580 to i64
  %582 = or disjoint i64 %31, %581
  %583 = zext nneg i32 %15 to i64
  %584 = getelementptr inbounds %struct.__half, ptr addrspace(1) %2, i64 %583
  %585 = extractelement <4 x float> %572, i64 0
  %586 = extractelement <4 x float> %572, i64 1
  %587 = extractelement <4 x float> %572, i64 2
  %588 = extractelement <4 x float> %572, i64 3
  %589 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %590 = fptrunc float %585 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %589), !noalias !42
  %591 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %592 = fptrunc float %586 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %591), !noalias !42
  %593 = bitcast half %590 to i16
  %594 = bitcast half %592 to i16
  %595 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %596 = fptrunc float %587 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %595), !noalias !47
  %597 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %598 = fptrunc float %588 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %597), !noalias !47
  %599 = bitcast half %596 to i16
  %600 = bitcast half %598 to i16
  %601 = zext i16 %600 to i64
  %602 = shl nuw i64 %601, 48
  %603 = zext i16 %599 to i64
  %604 = shl nuw nsw i64 %603, 32
  %605 = or disjoint i64 %602, %604
  %606 = zext i16 %594 to i64
  %607 = shl nuw nsw i64 %606, 16
  %608 = or disjoint i64 %605, %607
  %609 = zext i16 %593 to i64
  %610 = or disjoint i64 %608, %609
  %611 = getelementptr inbounds %struct.__half, ptr addrspace(1) %584, i64 %582
  store i64 %610, ptr addrspace(1) %611, align 8
  %612 = extractelement <4 x float> %573, i64 0
  %613 = extractelement <4 x float> %573, i64 1
  %614 = extractelement <4 x float> %573, i64 2
  %615 = extractelement <4 x float> %573, i64 3
  %616 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %617 = fptrunc float %612 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %616), !noalias !42
  %618 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %619 = fptrunc float %613 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %618), !noalias !42
  %620 = bitcast half %617 to i16
  %621 = bitcast half %619 to i16
  %622 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %623 = fptrunc float %614 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %622), !noalias !47
  %624 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %625 = fptrunc float %615 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %624), !noalias !47
  %626 = bitcast half %623 to i16
  %627 = bitcast half %625 to i16
  %628 = zext i16 %627 to i64
  %629 = shl nuw i64 %628, 48
  %630 = zext i16 %626 to i64
  %631 = shl nuw nsw i64 %630, 32
  %632 = or disjoint i64 %629, %631
  %633 = zext i16 %621 to i64
  %634 = shl nuw nsw i64 %633, 16
  %635 = or disjoint i64 %632, %634
  %636 = zext i16 %620 to i64
  %637 = or disjoint i64 %635, %636
  %638 = getelementptr inbounds i8, ptr addrspace(1) %611, i64 32
  store i64 %637, ptr addrspace(1) %638, align 8
  %639 = extractelement <4 x float> %574, i64 0
  %640 = extractelement <4 x float> %574, i64 1
  %641 = extractelement <4 x float> %574, i64 2
  %642 = extractelement <4 x float> %574, i64 3
  %643 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %644 = fptrunc float %639 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %643), !noalias !42
  %645 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %646 = fptrunc float %640 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %645), !noalias !42
  %647 = bitcast half %644 to i16
  %648 = bitcast half %646 to i16
  %649 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %650 = fptrunc float %641 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %649), !noalias !47
  %651 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %652 = fptrunc float %642 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %651), !noalias !47
  %653 = bitcast half %650 to i16
  %654 = bitcast half %652 to i16
  %655 = zext i16 %654 to i64
  %656 = shl nuw i64 %655, 48
  %657 = zext i16 %653 to i64
  %658 = shl nuw nsw i64 %657, 32
  %659 = or disjoint i64 %656, %658
  %660 = zext i16 %648 to i64
  %661 = shl nuw nsw i64 %660, 16
  %662 = or disjoint i64 %659, %661
  %663 = zext i16 %647 to i64
  %664 = or disjoint i64 %662, %663
  %665 = getelementptr inbounds i8, ptr addrspace(1) %611, i64 64
  store i64 %664, ptr addrspace(1) %665, align 8
  %666 = extractelement <4 x float> %575, i64 0
  %667 = extractelement <4 x float> %575, i64 1
  %668 = extractelement <4 x float> %575, i64 2
  %669 = extractelement <4 x float> %575, i64 3
  %670 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %671 = fptrunc float %666 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %670), !noalias !42
  %672 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %673 = fptrunc float %667 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %672), !noalias !42
  %674 = bitcast half %671 to i16
  %675 = bitcast half %673 to i16
  %676 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %677 = fptrunc float %668 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %676), !noalias !47
  %678 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %679 = fptrunc float %669 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %678), !noalias !47
  %680 = bitcast half %677 to i16
  %681 = bitcast half %679 to i16
  %682 = zext i16 %681 to i64
  %683 = shl nuw i64 %682, 48
  %684 = zext i16 %680 to i64
  %685 = shl nuw nsw i64 %684, 32
  %686 = or disjoint i64 %683, %685
  %687 = zext i16 %675 to i64
  %688 = shl nuw nsw i64 %687, 16
  %689 = or disjoint i64 %686, %688
  %690 = zext i16 %674 to i64
  %691 = or disjoint i64 %689, %690
  %692 = getelementptr inbounds i8, ptr addrspace(1) %611, i64 96
  store i64 %691, ptr addrspace(1) %692, align 8
  %693 = extractelement <4 x float> %576, i64 0
  %694 = extractelement <4 x float> %576, i64 1
  %695 = extractelement <4 x float> %576, i64 2
  %696 = extractelement <4 x float> %576, i64 3
  %697 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %698 = fptrunc float %693 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %697), !noalias !42
  %699 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %700 = fptrunc float %694 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %699), !noalias !42
  %701 = bitcast half %698 to i16
  %702 = bitcast half %700 to i16
  %703 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %704 = fptrunc float %695 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %703), !noalias !47
  %705 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %706 = fptrunc float %696 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %705), !noalias !47
  %707 = bitcast half %704 to i16
  %708 = bitcast half %706 to i16
  %709 = zext i16 %708 to i64
  %710 = shl nuw i64 %709, 48
  %711 = zext i16 %707 to i64
  %712 = shl nuw nsw i64 %711, 32
  %713 = or disjoint i64 %710, %712
  %714 = zext i16 %702 to i64
  %715 = shl nuw nsw i64 %714, 16
  %716 = or disjoint i64 %713, %715
  %717 = zext i16 %701 to i64
  %718 = or disjoint i64 %716, %717
  %719 = getelementptr inbounds i8, ptr addrspace(1) %611, i64 128
  store i64 %718, ptr addrspace(1) %719, align 8
  %720 = extractelement <4 x float> %577, i64 0
  %721 = extractelement <4 x float> %577, i64 1
  %722 = extractelement <4 x float> %577, i64 2
  %723 = extractelement <4 x float> %577, i64 3
  %724 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %725 = fptrunc float %720 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %724), !noalias !42
  %726 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %727 = fptrunc float %721 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %726), !noalias !42
  %728 = bitcast half %725 to i16
  %729 = bitcast half %727 to i16
  %730 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %731 = fptrunc float %722 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %730), !noalias !47
  %732 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %733 = fptrunc float %723 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %732), !noalias !47
  %734 = bitcast half %731 to i16
  %735 = bitcast half %733 to i16
  %736 = zext i16 %735 to i64
  %737 = shl nuw i64 %736, 48
  %738 = zext i16 %734 to i64
  %739 = shl nuw nsw i64 %738, 32
  %740 = or disjoint i64 %737, %739
  %741 = zext i16 %729 to i64
  %742 = shl nuw nsw i64 %741, 16
  %743 = or disjoint i64 %740, %742
  %744 = zext i16 %728 to i64
  %745 = or disjoint i64 %743, %744
  %746 = getelementptr inbounds i8, ptr addrspace(1) %611, i64 160
  store i64 %745, ptr addrspace(1) %746, align 8
  %747 = extractelement <4 x float> %578, i64 0
  %748 = extractelement <4 x float> %578, i64 1
  %749 = extractelement <4 x float> %578, i64 2
  %750 = extractelement <4 x float> %578, i64 3
  %751 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %752 = fptrunc float %747 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %751), !noalias !42
  %753 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %754 = fptrunc float %748 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %753), !noalias !42
  %755 = bitcast half %752 to i16
  %756 = bitcast half %754 to i16
  %757 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %758 = fptrunc float %749 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %757), !noalias !47
  %759 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %760 = fptrunc float %750 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %759), !noalias !47
  %761 = bitcast half %758 to i16
  %762 = bitcast half %760 to i16
  %763 = zext i16 %762 to i64
  %764 = shl nuw i64 %763, 48
  %765 = zext i16 %761 to i64
  %766 = shl nuw nsw i64 %765, 32
  %767 = or disjoint i64 %764, %766
  %768 = zext i16 %756 to i64
  %769 = shl nuw nsw i64 %768, 16
  %770 = or disjoint i64 %767, %769
  %771 = zext i16 %755 to i64
  %772 = or disjoint i64 %770, %771
  %773 = getelementptr inbounds i8, ptr addrspace(1) %611, i64 192
  store i64 %772, ptr addrspace(1) %773, align 8
  %774 = extractelement <4 x float> %579, i64 0
  %775 = extractelement <4 x float> %579, i64 1
  %776 = extractelement <4 x float> %579, i64 2
  %777 = extractelement <4 x float> %579, i64 3
  %778 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %779 = fptrunc float %774 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %778), !noalias !42
  %780 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !42
  %781 = fptrunc float %775 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %780), !noalias !42
  %782 = bitcast half %779 to i16
  %783 = bitcast half %781 to i16
  %784 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %785 = fptrunc float %776 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %784), !noalias !47
  %786 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !47
  %787 = fptrunc float %777 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %786), !noalias !47
  %788 = bitcast half %785 to i16
  %789 = bitcast half %787 to i16
  %790 = zext i16 %789 to i64
  %791 = shl nuw i64 %790, 48
  %792 = zext i16 %788 to i64
  %793 = shl nuw nsw i64 %792, 32
  %794 = or disjoint i64 %791, %793
  %795 = zext i16 %783 to i64
  %796 = shl nuw nsw i64 %795, 16
  %797 = or disjoint i64 %794, %796
  %798 = zext i16 %782 to i64
  %799 = or disjoint i64 %797, %798
  %800 = getelementptr inbounds i8, ptr addrspace(1) %611, i64 224
  store i64 %799, ptr addrspace(1) %800, align 8
  br label %801

801:                                              ; preds = %395, %5
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
