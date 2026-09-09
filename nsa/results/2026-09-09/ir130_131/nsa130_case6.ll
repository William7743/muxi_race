; ModuleID = '/data/nsa_codex_20260909_r1/results/ir130_131/nsa130_case6.bc'
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
  %6 = alloca [16 x %struct.__half], align 2, addrspace(5)
  call void @llvm.lifetime.start.p5(i64 32, ptr addrspace(5) %6)
  %7 = tail call noundef range(i32 0, 2147483647) i32 @llvm.mxc.block.id.y(), !range !16
  %8 = shl nsw i32 %7, 10
  %9 = tail call noundef range(i32 0, 2147483647) i32 @llvm.mxc.block.id.x(), !range !16
  %10 = add nuw nsw i32 %8, %9
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %11
  %13 = load i32, ptr addrspace(1) %12, align 4, !tbaa !17
  %14 = lshr i32 %9, 5
  %15 = icmp ugt i32 %13, %14
  br i1 %15, label %469, label %16

16:                                               ; preds = %5
  %17 = shl nuw nsw i32 %13, 5
  %18 = and i32 %17, 992
  %19 = tail call noundef range(i32 0, 1024) i32 @llvm.mxc.thread.id.x(), !range !26
  %20 = lshr i32 %19, 2
  %21 = and i32 %20, 252
  %22 = add nuw nsw i32 %18, %21
  %23 = icmp ugt i32 %22, %9
  %24 = select i1 %23, float 0xFFF0000000000000, float 0.000000e+00
  %25 = insertelement <4 x float> poison, float %24, i64 0
  %26 = icmp ult i32 %22, %9
  %27 = select i1 %26, float 0.000000e+00, float 0xFFF0000000000000
  %28 = insertelement <4 x float> %25, float %27, i64 1
  %29 = or disjoint i32 %22, 2
  %30 = icmp ugt i32 %29, %9
  %31 = select i1 %30, float 0xFFF0000000000000, float 0.000000e+00
  %32 = insertelement <4 x float> %28, float %31, i64 2
  %33 = or disjoint i32 %22, 3
  %34 = icmp ugt i32 %33, %9
  %35 = select i1 %34, float 0xFFF0000000000000, float 0.000000e+00
  %36 = insertelement <4 x float> %32, float %35, i64 3
  %37 = shl nsw i32 %7, 17
  %38 = shl i32 %13, 12
  %39 = and i32 %38, 126976
  %40 = shl nuw nsw i32 %19, 4
  %41 = and i32 %40, 16256
  %42 = shl nuw nsw i32 %19, 3
  %43 = and i32 %42, 56
  %44 = or disjoint i32 %41, %37
  %45 = or disjoint i32 %44, %43
  %46 = add nuw nsw i32 %45, %39
  %47 = zext nneg i32 %46 to i64
  %48 = getelementptr inbounds %struct.__half, ptr addrspace(4) %4, i64 %47
  call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) noundef align 16 dereferenceable(16) %6, ptr addrspace(4) noundef align 16 dereferenceable(16) %48, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %49 = getelementptr inbounds i8, ptr addrspace(4) %48, i64 4096
  %50 = getelementptr inbounds i8, ptr addrspace(5) %6, i32 16
  call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) noundef align 16 dereferenceable(16) %50, ptr addrspace(4) noundef align 16 dereferenceable(16) %49, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %51 = shl nsw i32 %7, 21
  %52 = shl nsw i32 %9, 11
  %53 = add nuw nsw i32 %51, %52
  %54 = add nuw nsw i32 %53, %41
  %55 = or disjoint i32 %54, %43
  %56 = and i32 %42, 8128
  %57 = and i32 %42, 32
  %58 = add nuw nsw i32 %57, %19
  %59 = and i32 %58, 32
  %60 = and i32 %42, 16
  %61 = add nuw nsw i32 %60, %19
  %62 = and i32 %61, 16
  %63 = mul nuw nsw i32 %19, 9
  %64 = and i32 %63, 8
  %65 = or disjoint i32 %59, %64
  %66 = or disjoint i32 %65, %56
  %67 = or disjoint i32 %66, %62
  %68 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %67
  %69 = shl nuw nsw i32 %19, 6
  %70 = and i32 %69, 960
  %71 = lshr i32 %19, 1
  %72 = lshr i32 %19, 5
  %73 = add nuw nsw i32 %72, %19
  %74 = shl nuw nsw i32 %73, 3
  %75 = and i32 %74, 8
  %76 = and i32 %20, 4
  %77 = or disjoint i32 %76, %70
  %78 = or disjoint i32 %77, %75
  %79 = and i32 %40, 15360
  %80 = or disjoint i32 %70, %79
  %81 = or disjoint i32 %80, %76
  %82 = or disjoint i32 %81, %75
  %83 = zext nneg i32 %55 to i64
  %84 = getelementptr inbounds %struct.__half, ptr addrspace(4) %3, i64 %83
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %68, ptr addrspace(4) noundef align 16 dereferenceable(16) %84, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  %85 = getelementptr inbounds %struct.__half, ptr addrspace(4) %1, i64 %47
  %86 = or disjoint i32 %56, %59
  %87 = or disjoint i32 %86, %62
  %88 = or disjoint i32 %87, %64
  %89 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %88
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %89, ptr addrspace(4) noundef align 16 dereferenceable(16) %85, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !30
  %90 = getelementptr inbounds i8, ptr addrspace(4) %85, i64 4096
  %91 = add nuw nsw i32 %56, 1024
  %92 = or disjoint i32 %91, %59
  %93 = or disjoint i32 %92, %62
  %94 = or disjoint i32 %93, %64
  %95 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %94
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %95, ptr addrspace(4) noundef align 16 dereferenceable(16) %90, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !30
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %96 = shl nuw nsw i32 %20, 5
  %97 = and i32 %96, 32
  %98 = shl nuw nsw i32 %71, 4
  %99 = and i32 %98, 16
  %100 = or disjoint i32 %78, %99
  %101 = or disjoint i32 %100, %97
  %102 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %101
  %103 = load <4 x half>, ptr addrspace(3) %102, align 8
  %104 = or disjoint i32 %82, %99
  %105 = or disjoint i32 %104, %97
  %106 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %105
  %107 = load <4 x half>, ptr addrspace(3) %106, align 8
  %108 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %107, <4 x half> %103, <4 x float> %36)
  %109 = xor i32 %99, 16
  %110 = or disjoint i32 %78, %109
  %111 = or disjoint i32 %110, %97
  %112 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %111
  %113 = load <4 x half>, ptr addrspace(3) %112, align 8
  %114 = or disjoint i32 %82, %109
  %115 = or disjoint i32 %114, %97
  %116 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %115
  %117 = load <4 x half>, ptr addrspace(3) %116, align 8
  %118 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %117, <4 x half> %113, <4 x float> %108)
  %119 = add nuw nsw i32 %20, 1
  %120 = shl nuw nsw i32 %119, 5
  %121 = and i32 %120, 32
  %122 = or disjoint i32 %100, %121
  %123 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %122
  %124 = load <4 x half>, ptr addrspace(3) %123, align 8
  %125 = or disjoint i32 %104, %121
  %126 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %125
  %127 = load <4 x half>, ptr addrspace(3) %126, align 8
  %128 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %127, <4 x half> %124, <4 x float> %118)
  %129 = or disjoint i32 %110, %121
  %130 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %129
  %131 = load <4 x half>, ptr addrspace(3) %130, align 8
  %132 = or disjoint i32 %114, %121
  %133 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %132
  %134 = load <4 x half>, ptr addrspace(3) %133, align 8
  %135 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %134, <4 x half> %131, <4 x float> %128)
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %136 = or disjoint i64 %83, 64
  %137 = getelementptr inbounds %struct.__half, ptr addrspace(4) %3, i64 %136
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %68, ptr addrspace(4) noundef align 16 dereferenceable(16) %137, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  %138 = or disjoint i64 %47, 64
  %139 = getelementptr inbounds %struct.__half, ptr addrspace(4) %1, i64 %138
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %89, ptr addrspace(4) noundef align 16 dereferenceable(16) %139, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !30
  %140 = getelementptr inbounds i8, ptr addrspace(4) %85, i64 4224
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %95, ptr addrspace(4) noundef align 16 dereferenceable(16) %140, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !30
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %141 = load <4 x half>, ptr addrspace(3) %102, align 8
  %142 = load <4 x half>, ptr addrspace(3) %106, align 8
  %143 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %142, <4 x half> %141, <4 x float> %135)
  %144 = load <4 x half>, ptr addrspace(3) %112, align 8
  %145 = load <4 x half>, ptr addrspace(3) %116, align 8
  %146 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %145, <4 x half> %144, <4 x float> %143)
  %147 = load <4 x half>, ptr addrspace(3) %123, align 8
  %148 = load <4 x half>, ptr addrspace(3) %126, align 8
  %149 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %148, <4 x half> %147, <4 x float> %146)
  %150 = load <4 x half>, ptr addrspace(3) %130, align 8
  %151 = load <4 x half>, ptr addrspace(3) %133, align 8
  %152 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %151, <4 x half> %150, <4 x float> %149)
  %153 = extractelement <4 x float> %152, i64 0
  %154 = tail call contract noundef float @llvm.maxnum.f32(float %153, float 0xFFF0000000000000)
  %155 = extractelement <4 x float> %152, i64 1
  %156 = tail call contract noundef float @llvm.maxnum.f32(float %154, float %155)
  %157 = extractelement <4 x float> %152, i64 2
  %158 = tail call contract noundef float @llvm.maxnum.f32(float %156, float %157)
  %159 = extractelement <4 x float> %152, i64 3
  %160 = tail call contract noundef float @llvm.maxnum.f32(float %158, float %159)
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %161 = getelementptr inbounds float, ptr addrspace(3) @buf_dyn_shmem, i32 %19
  store float %160, ptr addrspace(3) %161, align 4, !tbaa !31
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %162 = xor i32 %19, 64
  %163 = getelementptr inbounds float, ptr addrspace(3) @buf_dyn_shmem, i32 %162
  %164 = load float, ptr addrspace(3) %163, align 4, !tbaa !31
  %165 = tail call contract noundef float @llvm.maxnum.f32(float %160, float %164)
  %166 = bitcast float %165 to i32
  %167 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #11
  %168 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %167) #11
  %169 = xor i32 %168, 32
  %170 = and i32 %168, -64
  %171 = add nsw i32 %170, 64
  %172 = icmp slt i32 %169, %171
  %173 = select i1 %172, i32 %169, i32 %168
  %174 = shl i32 %173, 2
  %175 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %174, i32 %166)
  %176 = bitcast i32 %175 to float
  %177 = tail call contract noundef float @llvm.maxnum.f32(float %165, float %176)
  %178 = bitcast float %177 to i32
  %179 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #11
  %180 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %179) #11
  %181 = xor i32 %180, 16
  %182 = and i32 %180, -64
  %183 = add nsw i32 %182, 64
  %184 = icmp slt i32 %181, %183
  %185 = select i1 %184, i32 %181, i32 %180
  %186 = shl i32 %185, 2
  %187 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %186, i32 %178)
  %188 = bitcast i32 %187 to float
  %189 = tail call contract noundef float @llvm.maxnum.f32(float %177, float %188)
  %190 = fsub contract float %153, %189
  %191 = fmul contract float %190, 0x3FC0527DC0000000
  %192 = fcmp contract olt float %191, -1.260000e+02
  %193 = select contract i1 %192, float 6.400000e+01, float 0.000000e+00
  %194 = fadd contract float %191, %193
  %195 = tail call contract float @llvm.exp2.f32(float %194)
  %196 = select contract i1 %192, float 0x3BF0000000000000, float 1.000000e+00
  %197 = fmul contract float %196, %195
  %198 = fsub contract float %155, %189
  %199 = fmul contract float %198, 0x3FC0527DC0000000
  %200 = fcmp contract olt float %199, -1.260000e+02
  %201 = select contract i1 %200, float 6.400000e+01, float 0.000000e+00
  %202 = fadd contract float %199, %201
  %203 = tail call contract float @llvm.exp2.f32(float %202)
  %204 = select contract i1 %200, float 0x3BF0000000000000, float 1.000000e+00
  %205 = fmul contract float %204, %203
  %206 = fsub contract float %157, %189
  %207 = fmul contract float %206, 0x3FC0527DC0000000
  %208 = fcmp contract olt float %207, -1.260000e+02
  %209 = select contract i1 %208, float 6.400000e+01, float 0.000000e+00
  %210 = fadd contract float %207, %209
  %211 = tail call contract float @llvm.exp2.f32(float %210)
  %212 = select contract i1 %208, float 0x3BF0000000000000, float 1.000000e+00
  %213 = fmul contract float %212, %211
  %214 = fsub contract float %159, %189
  %215 = fmul contract float %214, 0x3FC0527DC0000000
  %216 = fcmp contract olt float %215, -1.260000e+02
  %217 = select contract i1 %216, float 6.400000e+01, float 0.000000e+00
  %218 = fadd contract float %215, %217
  %219 = tail call contract float @llvm.exp2.f32(float %218)
  %220 = select contract i1 %216, float 0x3BF0000000000000, float 1.000000e+00
  %221 = fmul contract float %220, %219
  %222 = fadd contract float %197, 0.000000e+00
  %223 = fadd contract float %222, %205
  %224 = fadd contract float %223, %213
  %225 = fadd contract float %224, %221
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  store float %225, ptr addrspace(3) %161, align 4, !tbaa !31
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %226 = load float, ptr addrspace(3) %163, align 4, !tbaa !31
  %227 = fadd contract float %225, %226
  %228 = bitcast float %227 to i32
  %229 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #11
  %230 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %229) #11
  %231 = xor i32 %230, 32
  %232 = and i32 %230, -64
  %233 = add nsw i32 %232, 64
  %234 = icmp slt i32 %231, %233
  %235 = select i1 %234, i32 %231, i32 %230
  %236 = shl i32 %235, 2
  %237 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %236, i32 %228)
  %238 = bitcast i32 %237 to float
  %239 = fadd contract float %227, %238
  %240 = bitcast float %239 to i32
  %241 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #11
  %242 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %241) #11
  %243 = xor i32 %242, 16
  %244 = and i32 %242, -64
  %245 = add nsw i32 %244, 64
  %246 = icmp slt i32 %243, %245
  %247 = select i1 %246, i32 %243, i32 %242
  %248 = shl i32 %247, 2
  %249 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %248, i32 %240)
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %250 = bitcast i32 %249 to float
  %251 = fadd contract float %239, %250
  %252 = fdiv contract float %221, %251
  %253 = fdiv contract float %213, %251
  %254 = fdiv contract float %205, %251
  %255 = fdiv contract float %197, %251
  %256 = fptrunc float %255 to half
  %257 = fptrunc float %254 to half
  %258 = fptrunc float %253 to half
  %259 = fptrunc float %252 to half
  %260 = shl nuw nsw i32 %19, 5
  %261 = and i32 %260, 480
  %262 = lshr i32 %19, 6
  %263 = add nuw nsw i32 %262, %20
  %264 = shl nuw nsw i32 %263, 4
  %265 = and i32 %264, 16
  %266 = add nuw nsw i32 %72, %71
  %267 = shl nuw nsw i32 %266, 3
  %268 = and i32 %267, 8
  %269 = or disjoint i32 %261, %265
  %270 = or disjoint i32 %269, %76
  %271 = or disjoint i32 %270, %268
  %272 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %271
  store half %256, ptr addrspace(3) %272, align 8
  %273 = getelementptr inbounds i8, ptr addrspace(3) %272, i32 2
  store half %257, ptr addrspace(3) %273, align 2
  %274 = getelementptr inbounds i8, ptr addrspace(3) %272, i32 4
  store half %258, ptr addrspace(3) %274, align 4
  %275 = getelementptr inbounds i8, ptr addrspace(3) %272, i32 6
  store half %259, ptr addrspace(3) %275, align 2
  %276 = and i32 %40, 768
  %277 = lshr i32 %19, 4
  %278 = add nuw nsw i32 %262, %277
  %279 = shl nuw nsw i32 %278, 5
  %280 = and i32 %279, 32
  %281 = and i32 %19, 7
  %282 = or disjoint i32 %64, %56
  %283 = or disjoint i32 %282, %59
  %284 = or disjoint i32 %283, %62
  %285 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %284
  %286 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %88
  call void @llvm.memcpy.p3.p5.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %286, ptr addrspace(5) noundef align 16 dereferenceable(16) %6, i64 16, i1 false), !tbaa.struct !27
  %287 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %94
  call void @llvm.memcpy.p3.p5.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %287, ptr addrspace(5) noundef align 16 dereferenceable(16) %50, i64 16, i1 false), !tbaa.struct !27
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %288 = shl nuw nsw i32 %20, 4
  %289 = and i32 %288, 16
  %290 = or disjoint i32 %261, %289
  %291 = or disjoint i32 %290, %76
  %292 = or disjoint i32 %291, %268
  %293 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %292
  %294 = load <4 x half>, ptr addrspace(3) %293, align 8
  %295 = or disjoint i32 %276, %280
  %296 = and i32 %19, 8
  %297 = or disjoint i32 %295, %296
  %298 = or disjoint i32 %297, %281
  %299 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %298
  %300 = load half, ptr addrspace(3) %299, align 2, !tbaa !33
  %301 = insertelement <4 x half> poison, half %300, i64 0
  %302 = or disjoint i32 %296, %295
  %303 = or disjoint i32 %302, %281
  %304 = xor i32 %303, 8
  %305 = or disjoint i32 %304, 64
  %306 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %305
  %307 = load half, ptr addrspace(3) %306, align 2, !tbaa !33
  %308 = insertelement <4 x half> %301, half %307, i64 1
  %309 = or disjoint i32 %298, 144
  %310 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %309
  %311 = load half, ptr addrspace(3) %310, align 2, !tbaa !33
  %312 = insertelement <4 x half> %308, half %311, i64 2
  %313 = or disjoint i32 %304, 208
  %314 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %313
  %315 = load half, ptr addrspace(3) %314, align 2, !tbaa !33
  %316 = insertelement <4 x half> %312, half %315, i64 3
  %317 = or disjoint i32 %298, 16
  %318 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %317
  %319 = load half, ptr addrspace(3) %318, align 2, !tbaa !33
  %320 = insertelement <4 x half> poison, half %319, i64 0
  %321 = or disjoint i32 %304, 80
  %322 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %321
  %323 = load half, ptr addrspace(3) %322, align 2, !tbaa !33
  %324 = insertelement <4 x half> %320, half %323, i64 1
  %325 = or disjoint i32 %298, 128
  %326 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %325
  %327 = load half, ptr addrspace(3) %326, align 2, !tbaa !33
  %328 = insertelement <4 x half> %324, half %327, i64 2
  %329 = or disjoint i32 %304, 192
  %330 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %329
  %331 = load half, ptr addrspace(3) %330, align 2, !tbaa !33
  %332 = insertelement <4 x half> %328, half %331, i64 3
  %333 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %316, <4 x half> %294, <4 x float> zeroinitializer)
  %334 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %332, <4 x half> %294, <4 x float> zeroinitializer)
  %335 = shl nuw nsw i32 %119, 4
  %336 = and i32 %335, 16
  %337 = or disjoint i32 %261, %336
  %338 = or disjoint i32 %337, %76
  %339 = or disjoint i32 %338, %268
  %340 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %339
  %341 = load <4 x half>, ptr addrspace(3) %340, align 8
  %342 = or disjoint i32 %298, 1024
  %343 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %342
  %344 = load half, ptr addrspace(3) %343, align 2, !tbaa !33
  %345 = insertelement <4 x half> poison, half %344, i64 0
  %346 = or disjoint i32 %304, 1088
  %347 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %346
  %348 = load half, ptr addrspace(3) %347, align 2, !tbaa !33
  %349 = insertelement <4 x half> %345, half %348, i64 1
  %350 = or disjoint i32 %298, 1168
  %351 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %350
  %352 = load half, ptr addrspace(3) %351, align 2, !tbaa !33
  %353 = insertelement <4 x half> %349, half %352, i64 2
  %354 = or disjoint i32 %304, 1232
  %355 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %354
  %356 = load half, ptr addrspace(3) %355, align 2, !tbaa !33
  %357 = insertelement <4 x half> %353, half %356, i64 3
  %358 = or disjoint i32 %298, 1040
  %359 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %358
  %360 = load half, ptr addrspace(3) %359, align 2, !tbaa !33
  %361 = insertelement <4 x half> poison, half %360, i64 0
  %362 = or disjoint i32 %304, 1104
  %363 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %362
  %364 = load half, ptr addrspace(3) %363, align 2, !tbaa !33
  %365 = insertelement <4 x half> %361, half %364, i64 1
  %366 = or disjoint i32 %298, 1152
  %367 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %366
  %368 = load half, ptr addrspace(3) %367, align 2, !tbaa !33
  %369 = insertelement <4 x half> %365, half %368, i64 2
  %370 = or disjoint i32 %304, 1216
  %371 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %370
  %372 = load half, ptr addrspace(3) %371, align 2, !tbaa !33
  %373 = insertelement <4 x half> %369, half %372, i64 3
  %374 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %357, <4 x half> %341, <4 x float> %333)
  %375 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %373, <4 x half> %341, <4 x float> %334)
  %376 = or disjoint i32 %45, 64
  %377 = add nuw nsw i32 %376, %39
  %378 = zext nneg i32 %377 to i64
  %379 = shl nuw nsw i32 %263, 5
  %380 = and i32 %379, 32
  %381 = or disjoint i32 %70, %380
  %382 = or disjoint i32 %381, %75
  %383 = extractelement <4 x float> %375, i64 3
  %384 = extractelement <4 x float> %375, i64 2
  %385 = extractelement <4 x float> %375, i64 1
  %386 = extractelement <4 x float> %375, i64 0
  %387 = extractelement <4 x float> %374, i64 3
  %388 = extractelement <4 x float> %374, i64 2
  %389 = extractelement <4 x float> %374, i64 1
  %390 = extractelement <4 x float> %374, i64 0
  %391 = fptrunc float %390 to half
  %392 = fptrunc float %389 to half
  %393 = fptrunc float %388 to half
  %394 = fptrunc float %387 to half
  %395 = fptrunc float %386 to half
  %396 = fptrunc float %385 to half
  %397 = fptrunc float %384 to half
  %398 = fptrunc float %383 to half
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  %399 = or disjoint i32 %382, %99
  %400 = or disjoint i32 %399, %76
  %401 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %400
  store half %391, ptr addrspace(3) %401, align 8
  %402 = getelementptr inbounds i8, ptr addrspace(3) %401, i32 2
  store half %392, ptr addrspace(3) %402, align 2
  %403 = getelementptr inbounds i8, ptr addrspace(3) %401, i32 4
  store half %393, ptr addrspace(3) %403, align 4
  %404 = getelementptr inbounds i8, ptr addrspace(3) %401, i32 6
  store half %394, ptr addrspace(3) %404, align 2
  %405 = or disjoint i32 %382, %109
  %406 = or disjoint i32 %405, %76
  %407 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %406
  store half %395, ptr addrspace(3) %407, align 8
  %408 = getelementptr inbounds i8, ptr addrspace(3) %407, i32 2
  store half %396, ptr addrspace(3) %408, align 2
  %409 = getelementptr inbounds i8, ptr addrspace(3) %407, i32 4
  store half %397, ptr addrspace(3) %409, align 4
  %410 = getelementptr inbounds i8, ptr addrspace(3) %407, i32 6
  store half %398, ptr addrspace(3) %410, align 2
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %411 = getelementptr inbounds %struct.__half, ptr addrspace(1) %2, i64 %83
  tail call void @llvm.memcpy.p1.p3.i64(ptr addrspace(1) noundef align 16 dereferenceable(16) %411, ptr addrspace(3) noundef align 16 dereferenceable(16) %285, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !34
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  %412 = getelementptr inbounds %struct.__half, ptr addrspace(4) %4, i64 %378
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %286, ptr addrspace(4) noundef align 16 dereferenceable(16) %412, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %413 = getelementptr inbounds i8, ptr addrspace(4) %412, i64 4096
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %287, ptr addrspace(4) noundef align 16 dereferenceable(16) %413, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %414 = load <4 x half>, ptr addrspace(3) %293, align 8
  %415 = load half, ptr addrspace(3) %299, align 2, !tbaa !33
  %416 = insertelement <4 x half> poison, half %415, i64 0
  %417 = load half, ptr addrspace(3) %306, align 2, !tbaa !33
  %418 = insertelement <4 x half> %416, half %417, i64 1
  %419 = load half, ptr addrspace(3) %310, align 2, !tbaa !33
  %420 = insertelement <4 x half> %418, half %419, i64 2
  %421 = load half, ptr addrspace(3) %314, align 2, !tbaa !33
  %422 = insertelement <4 x half> %420, half %421, i64 3
  %423 = load half, ptr addrspace(3) %318, align 2, !tbaa !33
  %424 = insertelement <4 x half> poison, half %423, i64 0
  %425 = load half, ptr addrspace(3) %322, align 2, !tbaa !33
  %426 = insertelement <4 x half> %424, half %425, i64 1
  %427 = load half, ptr addrspace(3) %326, align 2, !tbaa !33
  %428 = insertelement <4 x half> %426, half %427, i64 2
  %429 = load half, ptr addrspace(3) %330, align 2, !tbaa !33
  %430 = insertelement <4 x half> %428, half %429, i64 3
  %431 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %422, <4 x half> %414, <4 x float> zeroinitializer)
  %432 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %430, <4 x half> %414, <4 x float> zeroinitializer)
  %433 = load <4 x half>, ptr addrspace(3) %340, align 8
  %434 = load half, ptr addrspace(3) %343, align 2, !tbaa !33
  %435 = insertelement <4 x half> poison, half %434, i64 0
  %436 = load half, ptr addrspace(3) %347, align 2, !tbaa !33
  %437 = insertelement <4 x half> %435, half %436, i64 1
  %438 = load half, ptr addrspace(3) %351, align 2, !tbaa !33
  %439 = insertelement <4 x half> %437, half %438, i64 2
  %440 = load half, ptr addrspace(3) %355, align 2, !tbaa !33
  %441 = insertelement <4 x half> %439, half %440, i64 3
  %442 = load half, ptr addrspace(3) %359, align 2, !tbaa !33
  %443 = insertelement <4 x half> poison, half %442, i64 0
  %444 = load half, ptr addrspace(3) %363, align 2, !tbaa !33
  %445 = insertelement <4 x half> %443, half %444, i64 1
  %446 = load half, ptr addrspace(3) %367, align 2, !tbaa !33
  %447 = insertelement <4 x half> %445, half %446, i64 2
  %448 = load half, ptr addrspace(3) %371, align 2, !tbaa !33
  %449 = insertelement <4 x half> %447, half %448, i64 3
  %450 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %441, <4 x half> %433, <4 x float> %431)
  %451 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %449, <4 x half> %433, <4 x float> %432)
  %452 = extractelement <4 x float> %451, i64 3
  %453 = extractelement <4 x float> %451, i64 2
  %454 = extractelement <4 x float> %451, i64 1
  %455 = extractelement <4 x float> %451, i64 0
  %456 = extractelement <4 x float> %450, i64 3
  %457 = extractelement <4 x float> %450, i64 2
  %458 = extractelement <4 x float> %450, i64 1
  %459 = extractelement <4 x float> %450, i64 0
  %460 = fptrunc float %459 to half
  %461 = fptrunc float %458 to half
  %462 = fptrunc float %457 to half
  %463 = fptrunc float %456 to half
  %464 = fptrunc float %455 to half
  %465 = fptrunc float %454 to half
  %466 = fptrunc float %453 to half
  %467 = fptrunc float %452 to half
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  store half %460, ptr addrspace(3) %401, align 8
  store half %461, ptr addrspace(3) %402, align 2
  store half %462, ptr addrspace(3) %403, align 4
  store half %463, ptr addrspace(3) %404, align 2
  store half %464, ptr addrspace(3) %407, align 8
  store half %465, ptr addrspace(3) %408, align 2
  store half %466, ptr addrspace(3) %409, align 4
  store half %467, ptr addrspace(3) %410, align 2
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %468 = getelementptr inbounds %struct.__half, ptr addrspace(1) %2, i64 %136
  tail call void @llvm.memcpy.p1.p3.i64(ptr addrspace(1) noundef align 16 dereferenceable(16) %468, ptr addrspace(3) noundef align 16 dereferenceable(16) %285, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !34
  br label %469

469:                                              ; preds = %16, %5
  call void @llvm.lifetime.end.p5(i64 32, ptr addrspace(5) %6)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p5(i64 immarg, ptr addrspace(5) nocapture) #6

; Function Attrs: nounwind speculatable willreturn memory(none)
declare i32 @llvm.mxc.thread.id.x() #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg) #7

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg) #7

; Function Attrs: convergent nounwind willreturn
declare void @llvm.mxc.barrier() #8

; Function Attrs: convergent nounwind willreturn memory(none)
declare <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>) #9

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #10

; Function Attrs: convergent nounwind willreturn memory(none)
declare i32 @llvm.mxc.mbcnt.lo(i32, i32) #9

; Function Attrs: convergent nounwind willreturn memory(none)
declare i32 @llvm.mxc.mbcnt.hi(i32, i32) #9

; Function Attrs: convergent nounwind willreturn memory(none)
declare i32 @llvm.mxc.bsm.bpermute(i32, i32) #9

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp2.f32(float) #10

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p3.p5.i64(ptr addrspace(3) noalias nocapture writeonly, ptr addrspace(5) noalias nocapture readonly, i64, i1 immarg) #7

; Function Attrs: convergent nounwind willreturn
declare void @llvm.mxc.barrier.warp() #8

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p1.p3.i64(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i64, i1 immarg) #7

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p5(i64 immarg, ptr addrspace(5) nocapture) #6

attributes #0 = { mustprogress noreturn nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" "disable-promote-alloca-to-bsm"="true" "disable-promote-alloca-to-vector"="false" "enable-ldg-bsm-opt"="false" "fixed-function-abi"="true" "metaxgpu-bsm-direct-address"="true" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="512" "metaxgpu-new-streg-abi"="false" "metaxgpu-pk-fma"="false" "metaxgpu-resource-usage"="false" "metaxgpu-sched-select"="default" "metaxgpu-use-dim-intrinsic"="false" "no-trapping-math"="true" "prec-div"="false" "prec-sqrt"="false" "scalarize-global-loads"="true" "shfl-combine"="true" "stack-protector-buffer-size"="8" "target-cpu"="xcore1000" "target-features"="+xcore1000" }
attributes #1 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #2 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" "disable-promote-alloca-to-bsm"="true" "disable-promote-alloca-to-vector"="false" "enable-ldg-bsm-opt"="false" "fixed-function-abi"="true" "metaxgpu-bsm-direct-address"="true" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="512" "metaxgpu-new-streg-abi"="false" "metaxgpu-pk-fma"="false" "metaxgpu-resource-usage"="false" "metaxgpu-sched-select"="default" "metaxgpu-use-dim-intrinsic"="false" "no-trapping-math"="true" "prec-div"="false" "prec-sqrt"="false" "scalarize-global-loads"="true" "shfl-combine"="true" "stack-protector-buffer-size"="8" "target-cpu"="xcore1000" "target-features"="+xcore1000" }
attributes #3 = { nounwind speculatable willreturn memory(none) }
attributes #4 = { nounwind willreturn }
attributes #5 = { convergent mustprogress norecurse nounwind willreturn "denormal-fp-math-f32"="preserve-sign,preserve-sign" "disable-promote-alloca-to-bsm"="true" "disable-promote-alloca-to-vector"="false" "enable-ldg-bsm-opt"="false" "fixed-function-abi"="true" "metaxgpu-bsm-direct-address"="true" "metaxgpu-implicitarg-num-bytes"="80" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="128" "metaxgpu-min-blocks"="1" "metaxgpu-new-streg-abi"="false" "metaxgpu-pk-fma"="false" "metaxgpu-resource-usage"="false" "metaxgpu-sched-select"="default" "metaxgpu-use-dim-intrinsic"="false" "no-trapping-math"="true" "prec-div"="false" "prec-sqrt"="false" "scalarize-global-loads"="true" "shfl-combine"="true" "stack-protector-buffer-size"="8" "target-cpu"="xcore1000" "target-features"="+xcore1000" "uniform-work-group-size"="true" }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #8 = { convergent nounwind willreturn }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
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
!28 = !{i32 -1, i32 4, i32 -1, i32 -1}
!29 = !{i32 -1, i32 3, i32 -1, i32 -1}
!30 = !{i32 -1, i32 1, i32 -1, i32 -1}
!31 = !{!32, !32, i64 0}
!32 = !{!"float", !5, i64 0}
!33 = !{!12, !12, i64 0}
!34 = !{i32 2, i32 -1, i32 -1, i32 -1}
