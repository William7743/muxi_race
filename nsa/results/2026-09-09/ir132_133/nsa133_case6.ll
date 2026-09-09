; ModuleID = '/data/nsa_codex_20260909_r1/results/ir132_133/nsa133_case6.bc'
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
  %6 = tail call noundef range(i32 0, 2147483647) i32 @llvm.mxc.block.id.y(), !range !16
  %7 = shl nsw i32 %6, 10
  %8 = tail call noundef range(i32 0, 2147483647) i32 @llvm.mxc.block.id.x(), !range !16
  %9 = add nuw nsw i32 %7, %8
  %10 = zext nneg i32 %9 to i64
  %11 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %10
  %12 = load i32, ptr addrspace(1) %11, align 4, !tbaa !17
  %13 = shl nsw i32 %12, 5
  %14 = icmp sgt i32 %13, %8
  br i1 %14, label %569, label %15

15:                                               ; preds = %5
  %16 = tail call noundef range(i32 0, 1024) i32 @llvm.mxc.thread.id.x(), !range !26
  %17 = lshr i32 %16, 2
  %18 = and i32 %17, 252
  %19 = add nsw i32 %13, %18
  %20 = icmp sgt i32 %19, %8
  %21 = select i1 %20, float 0xFFF0000000000000, float 0.000000e+00
  %22 = insertelement <4 x float> poison, float %21, i64 0
  %23 = icmp slt i32 %19, %8
  %24 = select i1 %23, float 0.000000e+00, float 0xFFF0000000000000
  %25 = insertelement <4 x float> %22, float %24, i64 1
  %26 = or disjoint i32 %19, 2
  %27 = icmp sgt i32 %26, %8
  %28 = select i1 %27, float 0xFFF0000000000000, float 0.000000e+00
  %29 = insertelement <4 x float> %25, float %28, i64 2
  %30 = or disjoint i32 %19, 3
  %31 = icmp sgt i32 %30, %8
  %32 = select i1 %31, float 0xFFF0000000000000, float 0.000000e+00
  %33 = insertelement <4 x float> %29, float %32, i64 3
  %34 = zext nneg i32 %6 to i64
  %35 = shl nuw nsw i64 %34, 17
  %36 = shl nuw nsw i32 %16, 4
  %37 = and i32 %36, 16256
  %38 = zext nneg i32 %37 to i64
  %39 = sext i32 %13 to i64
  %40 = or disjoint i64 %35, %38
  %41 = shl nuw nsw i32 %16, 3
  %42 = and i32 %41, 56
  %43 = zext nneg i32 %42 to i64
  %44 = or disjoint i64 %40, %43
  %45 = getelementptr %struct.__half, ptr addrspace(4) %4, i64 %44
  %46 = shl nsw i64 %39, 8
  %47 = getelementptr i8, ptr addrspace(4) %45, i64 %46
  %48 = load i32, ptr addrspace(4) %47, align 16, !tbaa !17
  %49 = getelementptr inbounds i8, ptr addrspace(4) %47, i64 4
  %50 = load i32, ptr addrspace(4) %49, align 4, !tbaa !17
  %51 = getelementptr inbounds i8, ptr addrspace(4) %47, i64 8
  %52 = load i32, ptr addrspace(4) %51, align 8, !tbaa !17
  %53 = getelementptr inbounds i8, ptr addrspace(4) %47, i64 12
  %54 = load i32, ptr addrspace(4) %53, align 4, !tbaa !17
  %55 = getelementptr i8, ptr addrspace(4) %47, i64 4096
  %56 = load i32, ptr addrspace(4) %55, align 16, !tbaa !17
  %57 = getelementptr i8, ptr addrspace(4) %47, i64 4100
  %58 = load i32, ptr addrspace(4) %57, align 4, !tbaa !17
  %59 = getelementptr i8, ptr addrspace(4) %47, i64 4104
  %60 = load i32, ptr addrspace(4) %59, align 8, !tbaa !17
  %61 = getelementptr i8, ptr addrspace(4) %47, i64 4108
  %62 = load i32, ptr addrspace(4) %61, align 4, !tbaa !17
  %63 = shl nsw i32 %6, 21
  %64 = shl nsw i32 %8, 11
  %65 = add nuw nsw i32 %63, %64
  %66 = add nuw nsw i32 %65, %37
  %67 = or disjoint i32 %66, %42
  %68 = and i32 %41, 8128
  %69 = and i32 %41, 32
  %70 = add nuw nsw i32 %69, %16
  %71 = and i32 %70, 32
  %72 = and i32 %41, 16
  %73 = add nuw nsw i32 %72, %16
  %74 = and i32 %73, 16
  %75 = mul nuw nsw i32 %16, 9
  %76 = and i32 %75, 8
  %77 = or disjoint i32 %71, %76
  %78 = or disjoint i32 %77, %68
  %79 = or disjoint i32 %78, %74
  %80 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %79
  %81 = shl nuw nsw i32 %16, 6
  %82 = and i32 %81, 960
  %83 = lshr i32 %16, 1
  %84 = lshr i32 %16, 5
  %85 = add nuw nsw i32 %84, %16
  %86 = shl nuw nsw i32 %85, 3
  %87 = and i32 %86, 8
  %88 = and i32 %17, 4
  %89 = or disjoint i32 %88, %82
  %90 = or disjoint i32 %89, %87
  %91 = and i32 %36, 15360
  %92 = or disjoint i32 %82, %91
  %93 = or disjoint i32 %92, %88
  %94 = or disjoint i32 %93, %87
  %95 = zext nneg i32 %67 to i64
  %96 = getelementptr inbounds %struct.__half, ptr addrspace(4) %3, i64 %95
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %80, ptr addrspace(4) noundef align 16 dereferenceable(16) %96, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %97 = getelementptr %struct.__half, ptr addrspace(4) %1, i64 %44
  %98 = getelementptr i8, ptr addrspace(4) %97, i64 %46
  %99 = or disjoint i32 %68, %71
  %100 = or disjoint i32 %99, %74
  %101 = or disjoint i32 %100, %76
  %102 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %101
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %102, ptr addrspace(4) noundef align 16 dereferenceable(16) %98, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  %103 = getelementptr i8, ptr addrspace(4) %98, i64 4096
  %104 = add nuw nsw i32 %68, 1024
  %105 = or disjoint i32 %104, %71
  %106 = or disjoint i32 %105, %74
  %107 = or disjoint i32 %106, %76
  %108 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %107
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %108, ptr addrspace(4) noundef align 16 dereferenceable(16) %103, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %109 = shl nuw nsw i32 %17, 5
  %110 = and i32 %109, 32
  %111 = shl nuw nsw i32 %83, 4
  %112 = and i32 %111, 16
  %113 = or disjoint i32 %90, %112
  %114 = or disjoint i32 %113, %110
  %115 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %114
  %116 = load <4 x half>, ptr addrspace(3) %115, align 8
  %117 = or disjoint i32 %94, %112
  %118 = or disjoint i32 %117, %110
  %119 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %118
  %120 = load <4 x half>, ptr addrspace(3) %119, align 8
  %121 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %120, <4 x half> %116, <4 x float> %33)
  %122 = xor i32 %112, 16
  %123 = or disjoint i32 %90, %122
  %124 = or disjoint i32 %123, %110
  %125 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %124
  %126 = load <4 x half>, ptr addrspace(3) %125, align 8
  %127 = or disjoint i32 %94, %122
  %128 = or disjoint i32 %127, %110
  %129 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %128
  %130 = load <4 x half>, ptr addrspace(3) %129, align 8
  %131 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %130, <4 x half> %126, <4 x float> %121)
  %132 = add nuw nsw i32 %17, 1
  %133 = shl nuw nsw i32 %132, 5
  %134 = and i32 %133, 32
  %135 = or disjoint i32 %113, %134
  %136 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %135
  %137 = load <4 x half>, ptr addrspace(3) %136, align 8
  %138 = or disjoint i32 %117, %134
  %139 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %138
  %140 = load <4 x half>, ptr addrspace(3) %139, align 8
  %141 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %140, <4 x half> %137, <4 x float> %131)
  %142 = or disjoint i32 %123, %134
  %143 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %142
  %144 = load <4 x half>, ptr addrspace(3) %143, align 8
  %145 = or disjoint i32 %127, %134
  %146 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %145
  %147 = load <4 x half>, ptr addrspace(3) %146, align 8
  %148 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %147, <4 x half> %144, <4 x float> %141)
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %149 = or disjoint i64 %95, 64
  %150 = getelementptr inbounds %struct.__half, ptr addrspace(4) %3, i64 %149
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %80, ptr addrspace(4) noundef align 16 dereferenceable(16) %150, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %151 = getelementptr i8, ptr addrspace(4) %98, i64 128
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %102, ptr addrspace(4) noundef align 16 dereferenceable(16) %151, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  %152 = getelementptr i8, ptr addrspace(4) %98, i64 4224
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %108, ptr addrspace(4) noundef align 16 dereferenceable(16) %152, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %153 = load <4 x half>, ptr addrspace(3) %115, align 8
  %154 = load <4 x half>, ptr addrspace(3) %119, align 8
  %155 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %154, <4 x half> %153, <4 x float> %148)
  %156 = load <4 x half>, ptr addrspace(3) %125, align 8
  %157 = load <4 x half>, ptr addrspace(3) %129, align 8
  %158 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %157, <4 x half> %156, <4 x float> %155)
  %159 = load <4 x half>, ptr addrspace(3) %136, align 8
  %160 = load <4 x half>, ptr addrspace(3) %139, align 8
  %161 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %160, <4 x half> %159, <4 x float> %158)
  %162 = load <4 x half>, ptr addrspace(3) %143, align 8
  %163 = load <4 x half>, ptr addrspace(3) %146, align 8
  %164 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %163, <4 x half> %162, <4 x float> %161)
  %165 = extractelement <4 x float> %164, i64 0
  %166 = tail call contract noundef float @llvm.maxnum.f32(float %165, float 0xFFF0000000000000)
  %167 = extractelement <4 x float> %164, i64 1
  %168 = tail call contract noundef float @llvm.maxnum.f32(float %166, float %167)
  %169 = extractelement <4 x float> %164, i64 2
  %170 = tail call contract noundef float @llvm.maxnum.f32(float %168, float %169)
  %171 = extractelement <4 x float> %164, i64 3
  %172 = tail call contract noundef float @llvm.maxnum.f32(float %170, float %171)
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %173 = getelementptr inbounds float, ptr addrspace(3) @buf_dyn_shmem, i32 %16
  store float %172, ptr addrspace(3) %173, align 4, !tbaa !30
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %174 = xor i32 %16, 64
  %175 = getelementptr inbounds float, ptr addrspace(3) @buf_dyn_shmem, i32 %174
  %176 = load float, ptr addrspace(3) %175, align 4, !tbaa !30
  %177 = tail call contract noundef float @llvm.maxnum.f32(float %172, float %176)
  %178 = bitcast float %177 to i32
  %179 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #10
  %180 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %179) #10
  %181 = xor i32 %180, 32
  %182 = and i32 %180, -64
  %183 = add nsw i32 %182, 64
  %184 = icmp slt i32 %181, %183
  %185 = select i1 %184, i32 %181, i32 %180
  %186 = shl i32 %185, 2
  %187 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %186, i32 %178)
  %188 = bitcast i32 %187 to float
  %189 = tail call contract noundef float @llvm.maxnum.f32(float %177, float %188)
  %190 = bitcast float %189 to i32
  %191 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #10
  %192 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %191) #10
  %193 = xor i32 %192, 16
  %194 = and i32 %192, -64
  %195 = add nsw i32 %194, 64
  %196 = icmp slt i32 %193, %195
  %197 = select i1 %196, i32 %193, i32 %192
  %198 = shl i32 %197, 2
  %199 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %198, i32 %190)
  %200 = bitcast i32 %199 to float
  %201 = tail call contract noundef float @llvm.maxnum.f32(float %189, float %200)
  %202 = fsub contract float %165, %201
  %203 = fmul contract float %202, 0x3FC0527DC0000000
  %204 = fcmp contract olt float %203, -1.260000e+02
  %205 = select contract i1 %204, float 6.400000e+01, float 0.000000e+00
  %206 = fadd contract float %203, %205
  %207 = tail call contract float @llvm.exp2.f32(float %206)
  %208 = select contract i1 %204, float 0x3BF0000000000000, float 1.000000e+00
  %209 = fmul contract float %208, %207
  %210 = fsub contract float %167, %201
  %211 = fmul contract float %210, 0x3FC0527DC0000000
  %212 = fcmp contract olt float %211, -1.260000e+02
  %213 = select contract i1 %212, float 6.400000e+01, float 0.000000e+00
  %214 = fadd contract float %211, %213
  %215 = tail call contract float @llvm.exp2.f32(float %214)
  %216 = select contract i1 %212, float 0x3BF0000000000000, float 1.000000e+00
  %217 = fmul contract float %216, %215
  %218 = fsub contract float %169, %201
  %219 = fmul contract float %218, 0x3FC0527DC0000000
  %220 = fcmp contract olt float %219, -1.260000e+02
  %221 = select contract i1 %220, float 6.400000e+01, float 0.000000e+00
  %222 = fadd contract float %219, %221
  %223 = tail call contract float @llvm.exp2.f32(float %222)
  %224 = select contract i1 %220, float 0x3BF0000000000000, float 1.000000e+00
  %225 = fmul contract float %224, %223
  %226 = fsub contract float %171, %201
  %227 = fmul contract float %226, 0x3FC0527DC0000000
  %228 = fcmp contract olt float %227, -1.260000e+02
  %229 = select contract i1 %228, float 6.400000e+01, float 0.000000e+00
  %230 = fadd contract float %227, %229
  %231 = tail call contract float @llvm.exp2.f32(float %230)
  %232 = select contract i1 %228, float 0x3BF0000000000000, float 1.000000e+00
  %233 = fmul contract float %232, %231
  %234 = fadd contract float %209, 0.000000e+00
  %235 = fadd contract float %234, %217
  %236 = fadd contract float %235, %225
  %237 = fadd contract float %236, %233
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  store float %237, ptr addrspace(3) %173, align 4, !tbaa !30
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %238 = load float, ptr addrspace(3) %175, align 4, !tbaa !30
  %239 = fadd contract float %237, %238
  %240 = bitcast float %239 to i32
  %241 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #10
  %242 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %241) #10
  %243 = xor i32 %242, 32
  %244 = and i32 %242, -64
  %245 = add nsw i32 %244, 64
  %246 = icmp slt i32 %243, %245
  %247 = select i1 %246, i32 %243, i32 %242
  %248 = shl i32 %247, 2
  %249 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %248, i32 %240)
  %250 = bitcast i32 %249 to float
  %251 = fadd contract float %239, %250
  %252 = bitcast float %251 to i32
  %253 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #10
  %254 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %253) #10
  %255 = xor i32 %254, 16
  %256 = and i32 %254, -64
  %257 = add nsw i32 %256, 64
  %258 = icmp slt i32 %255, %257
  %259 = select i1 %258, i32 %255, i32 %254
  %260 = shl i32 %259, 2
  %261 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %260, i32 %252)
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %262 = bitcast i32 %261 to float
  %263 = fadd contract float %251, %262
  %264 = fdiv contract float %233, %263
  %265 = fdiv contract float %225, %263
  %266 = fdiv contract float %217, %263
  %267 = fdiv contract float %209, %263
  %268 = fptrunc float %267 to half
  %269 = fptrunc float %266 to half
  %270 = fptrunc float %265 to half
  %271 = fptrunc float %264 to half
  %272 = shl nuw nsw i32 %16, 5
  %273 = and i32 %272, 480
  %274 = lshr i32 %16, 6
  %275 = add nuw nsw i32 %274, %17
  %276 = shl nuw nsw i32 %275, 4
  %277 = and i32 %276, 16
  %278 = add nuw nsw i32 %84, %83
  %279 = shl nuw nsw i32 %278, 3
  %280 = and i32 %279, 8
  %281 = or disjoint i32 %273, %277
  %282 = or disjoint i32 %281, %88
  %283 = or disjoint i32 %282, %280
  %284 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %283
  store half %268, ptr addrspace(3) %284, align 8
  %285 = getelementptr inbounds i8, ptr addrspace(3) %284, i32 2
  store half %269, ptr addrspace(3) %285, align 2
  %286 = getelementptr inbounds i8, ptr addrspace(3) %284, i32 4
  store half %270, ptr addrspace(3) %286, align 4
  %287 = getelementptr inbounds i8, ptr addrspace(3) %284, i32 6
  store half %271, ptr addrspace(3) %287, align 2
  %288 = and i32 %36, 768
  %289 = lshr i32 %16, 4
  %290 = add nuw nsw i32 %274, %289
  %291 = shl nuw nsw i32 %290, 5
  %292 = and i32 %291, 32
  %293 = and i32 %16, 7
  %294 = or disjoint i32 %76, %68
  %295 = or disjoint i32 %294, %71
  %296 = or disjoint i32 %295, %74
  %297 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %296
  %298 = lshr i32 %16, 3
  %299 = add i32 %13, %298
  %300 = ashr i32 %299, 4
  %301 = or disjoint i64 %44, 64
  %302 = lshr i32 %48, 16
  %303 = trunc i32 %48 to i16
  %304 = trunc nuw i32 %302 to i16
  %305 = lshr i32 %50, 16
  %306 = trunc i32 %50 to i16
  %307 = trunc nuw i32 %305 to i16
  %308 = lshr i32 %52, 16
  %309 = trunc i32 %52 to i16
  %310 = trunc nuw i32 %308 to i16
  %311 = lshr i32 %54, 16
  %312 = trunc i32 %54 to i16
  %313 = trunc nuw i32 %311 to i16
  %314 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %101
  store i16 %303, ptr addrspace(3) %314, align 16
  %315 = getelementptr inbounds i8, ptr addrspace(3) %314, i32 2
  store i16 %304, ptr addrspace(3) %315, align 2, !tbaa !17
  %316 = getelementptr inbounds i8, ptr addrspace(3) %314, i32 4
  store i16 %306, ptr addrspace(3) %316, align 4
  %317 = getelementptr inbounds i8, ptr addrspace(3) %314, i32 6
  store i16 %307, ptr addrspace(3) %317, align 2, !tbaa !17
  %318 = getelementptr inbounds i8, ptr addrspace(3) %314, i32 8
  store i16 %309, ptr addrspace(3) %318, align 8
  %319 = getelementptr inbounds i8, ptr addrspace(3) %314, i32 10
  store i16 %310, ptr addrspace(3) %319, align 2, !tbaa !17
  %320 = getelementptr inbounds i8, ptr addrspace(3) %314, i32 12
  store i16 %312, ptr addrspace(3) %320, align 4
  %321 = getelementptr inbounds i8, ptr addrspace(3) %314, i32 14
  store i16 %313, ptr addrspace(3) %321, align 2, !tbaa !17
  %322 = lshr i32 %56, 16
  %323 = trunc i32 %56 to i16
  %324 = trunc nuw i32 %322 to i16
  %325 = lshr i32 %58, 16
  %326 = trunc i32 %58 to i16
  %327 = trunc nuw i32 %325 to i16
  %328 = lshr i32 %60, 16
  %329 = trunc i32 %60 to i16
  %330 = trunc nuw i32 %328 to i16
  %331 = lshr i32 %62, 16
  %332 = trunc i32 %62 to i16
  %333 = trunc nuw i32 %331 to i16
  %334 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %107
  store i16 %323, ptr addrspace(3) %334, align 16
  %335 = getelementptr inbounds i8, ptr addrspace(3) %334, i32 2
  store i16 %324, ptr addrspace(3) %335, align 2, !tbaa !17
  %336 = getelementptr inbounds i8, ptr addrspace(3) %334, i32 4
  store i16 %326, ptr addrspace(3) %336, align 4
  %337 = getelementptr inbounds i8, ptr addrspace(3) %334, i32 6
  store i16 %327, ptr addrspace(3) %337, align 2, !tbaa !17
  %338 = getelementptr inbounds i8, ptr addrspace(3) %334, i32 8
  store i16 %329, ptr addrspace(3) %338, align 8
  %339 = getelementptr inbounds i8, ptr addrspace(3) %334, i32 10
  store i16 %330, ptr addrspace(3) %339, align 2, !tbaa !17
  %340 = getelementptr inbounds i8, ptr addrspace(3) %334, i32 12
  store i16 %332, ptr addrspace(3) %340, align 4
  %341 = getelementptr inbounds i8, ptr addrspace(3) %334, i32 14
  store i16 %333, ptr addrspace(3) %341, align 2, !tbaa !17
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %342 = shl nuw nsw i32 %17, 4
  %343 = and i32 %342, 16
  %344 = or disjoint i32 %273, %343
  %345 = or disjoint i32 %344, %88
  %346 = or disjoint i32 %345, %280
  %347 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %346
  %348 = load <4 x half>, ptr addrspace(3) %347, align 8
  %349 = or disjoint i32 %288, %292
  %350 = and i32 %16, 8
  %351 = or disjoint i32 %349, %350
  %352 = or disjoint i32 %351, %293
  %353 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %352
  %354 = load half, ptr addrspace(3) %353, align 2, !tbaa !32
  %355 = insertelement <4 x half> poison, half %354, i64 0
  %356 = or disjoint i32 %350, %349
  %357 = or disjoint i32 %356, %293
  %358 = xor i32 %357, 8
  %359 = or disjoint i32 %358, 64
  %360 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %359
  %361 = load half, ptr addrspace(3) %360, align 2, !tbaa !32
  %362 = insertelement <4 x half> %355, half %361, i64 1
  %363 = or disjoint i32 %352, 144
  %364 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %363
  %365 = load half, ptr addrspace(3) %364, align 2, !tbaa !32
  %366 = insertelement <4 x half> %362, half %365, i64 2
  %367 = or disjoint i32 %358, 208
  %368 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %367
  %369 = load half, ptr addrspace(3) %368, align 2, !tbaa !32
  %370 = insertelement <4 x half> %366, half %369, i64 3
  %371 = or disjoint i32 %352, 16
  %372 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %371
  %373 = load half, ptr addrspace(3) %372, align 2, !tbaa !32
  %374 = insertelement <4 x half> poison, half %373, i64 0
  %375 = or disjoint i32 %358, 80
  %376 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %375
  %377 = load half, ptr addrspace(3) %376, align 2, !tbaa !32
  %378 = insertelement <4 x half> %374, half %377, i64 1
  %379 = or disjoint i32 %352, 128
  %380 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %379
  %381 = load half, ptr addrspace(3) %380, align 2, !tbaa !32
  %382 = insertelement <4 x half> %378, half %381, i64 2
  %383 = or disjoint i32 %358, 192
  %384 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %383
  %385 = load half, ptr addrspace(3) %384, align 2, !tbaa !32
  %386 = insertelement <4 x half> %382, half %385, i64 3
  %387 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %370, <4 x half> %348, <4 x float> zeroinitializer)
  %388 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %386, <4 x half> %348, <4 x float> zeroinitializer)
  %389 = shl nuw nsw i32 %132, 4
  %390 = and i32 %389, 16
  %391 = or disjoint i32 %273, %390
  %392 = or disjoint i32 %391, %88
  %393 = or disjoint i32 %392, %280
  %394 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %393
  %395 = load <4 x half>, ptr addrspace(3) %394, align 8
  %396 = or disjoint i32 %352, 1024
  %397 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %396
  %398 = load half, ptr addrspace(3) %397, align 2, !tbaa !32
  %399 = insertelement <4 x half> poison, half %398, i64 0
  %400 = or disjoint i32 %358, 1088
  %401 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %400
  %402 = load half, ptr addrspace(3) %401, align 2, !tbaa !32
  %403 = insertelement <4 x half> %399, half %402, i64 1
  %404 = or disjoint i32 %352, 1168
  %405 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %404
  %406 = load half, ptr addrspace(3) %405, align 2, !tbaa !32
  %407 = insertelement <4 x half> %403, half %406, i64 2
  %408 = or disjoint i32 %358, 1232
  %409 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %408
  %410 = load half, ptr addrspace(3) %409, align 2, !tbaa !32
  %411 = insertelement <4 x half> %407, half %410, i64 3
  %412 = or disjoint i32 %352, 1040
  %413 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %412
  %414 = load half, ptr addrspace(3) %413, align 2, !tbaa !32
  %415 = insertelement <4 x half> poison, half %414, i64 0
  %416 = or disjoint i32 %358, 1104
  %417 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %416
  %418 = load half, ptr addrspace(3) %417, align 2, !tbaa !32
  %419 = insertelement <4 x half> %415, half %418, i64 1
  %420 = or disjoint i32 %352, 1152
  %421 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %420
  %422 = load half, ptr addrspace(3) %421, align 2, !tbaa !32
  %423 = insertelement <4 x half> %419, half %422, i64 2
  %424 = or disjoint i32 %358, 1216
  %425 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %424
  %426 = load half, ptr addrspace(3) %425, align 2, !tbaa !32
  %427 = insertelement <4 x half> %423, half %426, i64 3
  %428 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %411, <4 x half> %395, <4 x float> %387)
  %429 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %427, <4 x half> %395, <4 x float> %388)
  %430 = icmp ult i32 %12, 32
  %431 = shl nuw nsw i32 %275, 5
  %432 = and i32 %431, 32
  %433 = or disjoint i32 %82, %432
  %434 = or disjoint i32 %433, %87
  %435 = extractelement <4 x float> %429, i64 3
  %436 = extractelement <4 x float> %429, i64 2
  %437 = extractelement <4 x float> %429, i64 1
  %438 = extractelement <4 x float> %429, i64 0
  %439 = extractelement <4 x float> %428, i64 3
  %440 = extractelement <4 x float> %428, i64 2
  %441 = extractelement <4 x float> %428, i64 1
  %442 = extractelement <4 x float> %428, i64 0
  %443 = fptrunc float %442 to half
  %444 = fptrunc float %441 to half
  %445 = fptrunc float %440 to half
  %446 = fptrunc float %439 to half
  %447 = fptrunc float %438 to half
  %448 = fptrunc float %437 to half
  %449 = fptrunc float %436 to half
  %450 = fptrunc float %435 to half
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  %451 = or disjoint i32 %434, %112
  %452 = or disjoint i32 %451, %88
  %453 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %452
  store half %443, ptr addrspace(3) %453, align 8
  %454 = getelementptr inbounds i8, ptr addrspace(3) %453, i32 2
  store half %444, ptr addrspace(3) %454, align 2
  %455 = getelementptr inbounds i8, ptr addrspace(3) %453, i32 4
  store half %445, ptr addrspace(3) %455, align 4
  %456 = getelementptr inbounds i8, ptr addrspace(3) %453, i32 6
  store half %446, ptr addrspace(3) %456, align 2
  %457 = or disjoint i32 %434, %122
  %458 = or disjoint i32 %457, %88
  %459 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %458
  store half %447, ptr addrspace(3) %459, align 8
  %460 = getelementptr inbounds i8, ptr addrspace(3) %459, i32 2
  store half %448, ptr addrspace(3) %460, align 2
  %461 = getelementptr inbounds i8, ptr addrspace(3) %459, i32 4
  store half %449, ptr addrspace(3) %461, align 4
  %462 = getelementptr inbounds i8, ptr addrspace(3) %459, i32 6
  store half %450, ptr addrspace(3) %462, align 2
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %463 = getelementptr inbounds %struct.__half, ptr addrspace(1) %2, i64 %95
  tail call void @llvm.memcpy.p1.p3.i64(ptr addrspace(1) noundef align 16 dereferenceable(16) %463, ptr addrspace(3) noundef align 16 dereferenceable(16) %297, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !33
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  br i1 %430, label %503, label %464

464:                                              ; preds = %15
  %465 = icmp slt i32 %300, 64
  %466 = icmp sgt i32 %299, -1
  %467 = and i1 %465, %466
  br i1 %467, label %468, label %478

468:                                              ; preds = %464
  %469 = getelementptr %struct.__half, ptr addrspace(4) %4, i64 %301
  %470 = getelementptr i8, ptr addrspace(4) %469, i64 %46
  %471 = load i32, ptr addrspace(4) %470, align 16, !tbaa !17
  %472 = getelementptr inbounds i8, ptr addrspace(4) %470, i64 4
  %473 = load i32, ptr addrspace(4) %472, align 4, !tbaa !17
  %474 = getelementptr inbounds i8, ptr addrspace(4) %470, i64 8
  %475 = load i32, ptr addrspace(4) %474, align 8, !tbaa !17
  %476 = getelementptr inbounds i8, ptr addrspace(4) %470, i64 12
  %477 = load i32, ptr addrspace(4) %476, align 4, !tbaa !17
  br label %478

478:                                              ; preds = %468, %464
  %479 = phi i32 [ %477, %468 ], [ 0, %464 ]
  %480 = phi i32 [ %475, %468 ], [ 0, %464 ]
  %481 = phi i32 [ %473, %468 ], [ 0, %464 ]
  %482 = phi i32 [ %471, %468 ], [ 0, %464 ]
  store i32 %482, ptr addrspace(3) %314, align 16, !tbaa !17
  store i32 %481, ptr addrspace(3) %316, align 4, !tbaa !17
  store i32 %480, ptr addrspace(3) %318, align 8, !tbaa !17
  store i32 %479, ptr addrspace(3) %320, align 4, !tbaa !17
  %483 = icmp slt i32 %300, 63
  %484 = add i32 %299, 16
  %485 = icmp sgt i32 %484, -1
  %486 = and i1 %483, %485
  br i1 %486, label %487, label %498

487:                                              ; preds = %478
  %488 = getelementptr %struct.__half, ptr addrspace(4) %4, i64 %301
  %489 = getelementptr i8, ptr addrspace(4) %488, i64 %46
  %490 = getelementptr i8, ptr addrspace(4) %489, i64 4096
  %491 = load i32, ptr addrspace(4) %490, align 16, !tbaa !17
  %492 = getelementptr i8, ptr addrspace(4) %489, i64 4100
  %493 = load i32, ptr addrspace(4) %492, align 4, !tbaa !17
  %494 = getelementptr i8, ptr addrspace(4) %489, i64 4104
  %495 = load i32, ptr addrspace(4) %494, align 8, !tbaa !17
  %496 = getelementptr i8, ptr addrspace(4) %489, i64 4108
  %497 = load i32, ptr addrspace(4) %496, align 4, !tbaa !17
  br label %498

498:                                              ; preds = %487, %478
  %499 = phi i32 [ %497, %487 ], [ 0, %478 ]
  %500 = phi i32 [ %495, %487 ], [ 0, %478 ]
  %501 = phi i32 [ %493, %487 ], [ 0, %478 ]
  %502 = phi i32 [ %491, %487 ], [ 0, %478 ]
  store i32 %502, ptr addrspace(3) %334, align 16, !tbaa !17
  store i32 %501, ptr addrspace(3) %336, align 4, !tbaa !17
  store i32 %500, ptr addrspace(3) %338, align 8, !tbaa !17
  store i32 %499, ptr addrspace(3) %340, align 4, !tbaa !17
  br label %513

503:                                              ; preds = %15
  %504 = shl nsw i32 %6, 17
  %505 = or disjoint i32 %504, %37
  %506 = or disjoint i32 %505, %42
  %507 = or disjoint i32 %506, 64
  %508 = shl nuw nsw i32 %12, 12
  %509 = add nuw nsw i32 %507, %508
  %510 = zext nneg i32 %509 to i64
  %511 = getelementptr inbounds %struct.__half, ptr addrspace(4) %4, i64 %510
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %314, ptr addrspace(4) noundef align 16 dereferenceable(16) %511, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !34
  %512 = getelementptr inbounds i8, ptr addrspace(4) %511, i64 4096
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %334, ptr addrspace(4) noundef align 16 dereferenceable(16) %512, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !34
  br label %513

513:                                              ; preds = %503, %498
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %514 = load <4 x half>, ptr addrspace(3) %347, align 8
  %515 = load half, ptr addrspace(3) %353, align 2, !tbaa !32
  %516 = insertelement <4 x half> poison, half %515, i64 0
  %517 = load half, ptr addrspace(3) %360, align 2, !tbaa !32
  %518 = insertelement <4 x half> %516, half %517, i64 1
  %519 = load half, ptr addrspace(3) %364, align 2, !tbaa !32
  %520 = insertelement <4 x half> %518, half %519, i64 2
  %521 = load half, ptr addrspace(3) %368, align 2, !tbaa !32
  %522 = insertelement <4 x half> %520, half %521, i64 3
  %523 = load half, ptr addrspace(3) %372, align 2, !tbaa !32
  %524 = insertelement <4 x half> poison, half %523, i64 0
  %525 = load half, ptr addrspace(3) %376, align 2, !tbaa !32
  %526 = insertelement <4 x half> %524, half %525, i64 1
  %527 = load half, ptr addrspace(3) %380, align 2, !tbaa !32
  %528 = insertelement <4 x half> %526, half %527, i64 2
  %529 = load half, ptr addrspace(3) %384, align 2, !tbaa !32
  %530 = insertelement <4 x half> %528, half %529, i64 3
  %531 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %522, <4 x half> %514, <4 x float> zeroinitializer)
  %532 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %530, <4 x half> %514, <4 x float> zeroinitializer)
  %533 = load <4 x half>, ptr addrspace(3) %394, align 8
  %534 = load half, ptr addrspace(3) %397, align 2, !tbaa !32
  %535 = insertelement <4 x half> poison, half %534, i64 0
  %536 = load half, ptr addrspace(3) %401, align 2, !tbaa !32
  %537 = insertelement <4 x half> %535, half %536, i64 1
  %538 = load half, ptr addrspace(3) %405, align 2, !tbaa !32
  %539 = insertelement <4 x half> %537, half %538, i64 2
  %540 = load half, ptr addrspace(3) %409, align 2, !tbaa !32
  %541 = insertelement <4 x half> %539, half %540, i64 3
  %542 = load half, ptr addrspace(3) %413, align 2, !tbaa !32
  %543 = insertelement <4 x half> poison, half %542, i64 0
  %544 = load half, ptr addrspace(3) %417, align 2, !tbaa !32
  %545 = insertelement <4 x half> %543, half %544, i64 1
  %546 = load half, ptr addrspace(3) %421, align 2, !tbaa !32
  %547 = insertelement <4 x half> %545, half %546, i64 2
  %548 = load half, ptr addrspace(3) %425, align 2, !tbaa !32
  %549 = insertelement <4 x half> %547, half %548, i64 3
  %550 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %541, <4 x half> %533, <4 x float> %531)
  %551 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %549, <4 x half> %533, <4 x float> %532)
  %552 = extractelement <4 x float> %551, i64 3
  %553 = extractelement <4 x float> %551, i64 2
  %554 = extractelement <4 x float> %551, i64 1
  %555 = extractelement <4 x float> %551, i64 0
  %556 = extractelement <4 x float> %550, i64 3
  %557 = extractelement <4 x float> %550, i64 2
  %558 = extractelement <4 x float> %550, i64 1
  %559 = extractelement <4 x float> %550, i64 0
  %560 = fptrunc float %559 to half
  %561 = fptrunc float %558 to half
  %562 = fptrunc float %557 to half
  %563 = fptrunc float %556 to half
  %564 = fptrunc float %555 to half
  %565 = fptrunc float %554 to half
  %566 = fptrunc float %553 to half
  %567 = fptrunc float %552 to half
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  store half %560, ptr addrspace(3) %453, align 8
  store half %561, ptr addrspace(3) %454, align 2
  store half %562, ptr addrspace(3) %455, align 4
  store half %563, ptr addrspace(3) %456, align 2
  store half %564, ptr addrspace(3) %459, align 8
  store half %565, ptr addrspace(3) %460, align 2
  store half %566, ptr addrspace(3) %461, align 4
  store half %567, ptr addrspace(3) %462, align 2
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %568 = getelementptr inbounds %struct.__half, ptr addrspace(1) %2, i64 %149
  tail call void @llvm.memcpy.p1.p3.i64(ptr addrspace(1) noundef align 16 dereferenceable(16) %568, ptr addrspace(3) noundef align 16 dereferenceable(16) %297, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !33
  br label %569

569:                                              ; preds = %513, %5
  ret void
}

; Function Attrs: nounwind speculatable willreturn memory(none)
declare i32 @llvm.mxc.thread.id.x() #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg) #6

; Function Attrs: convergent nounwind willreturn
declare void @llvm.mxc.barrier() #7

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

; Function Attrs: convergent nounwind willreturn
declare void @llvm.mxc.barrier.warp() #7

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p1.p3.i64(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i64, i1 immarg) #6

attributes #0 = { mustprogress noreturn nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" "disable-promote-alloca-to-bsm"="true" "disable-promote-alloca-to-vector"="false" "enable-ldg-bsm-opt"="false" "fixed-function-abi"="true" "metaxgpu-bsm-direct-address"="true" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="512" "metaxgpu-new-streg-abi"="false" "metaxgpu-pk-fma"="false" "metaxgpu-resource-usage"="false" "metaxgpu-sched-select"="default" "metaxgpu-use-dim-intrinsic"="false" "no-trapping-math"="true" "prec-div"="false" "prec-sqrt"="false" "scalarize-global-loads"="true" "shfl-combine"="true" "stack-protector-buffer-size"="8" "target-cpu"="xcore1000" "target-features"="+xcore1000" }
attributes #1 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #2 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" "disable-promote-alloca-to-bsm"="true" "disable-promote-alloca-to-vector"="false" "enable-ldg-bsm-opt"="false" "fixed-function-abi"="true" "metaxgpu-bsm-direct-address"="true" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="512" "metaxgpu-new-streg-abi"="false" "metaxgpu-pk-fma"="false" "metaxgpu-resource-usage"="false" "metaxgpu-sched-select"="default" "metaxgpu-use-dim-intrinsic"="false" "no-trapping-math"="true" "prec-div"="false" "prec-sqrt"="false" "scalarize-global-loads"="true" "shfl-combine"="true" "stack-protector-buffer-size"="8" "target-cpu"="xcore1000" "target-features"="+xcore1000" }
attributes #3 = { nounwind speculatable willreturn memory(none) }
attributes #4 = { nounwind willreturn }
attributes #5 = { convergent mustprogress norecurse nounwind willreturn "denormal-fp-math-f32"="preserve-sign,preserve-sign" "disable-promote-alloca-to-bsm"="true" "disable-promote-alloca-to-vector"="false" "enable-ldg-bsm-opt"="false" "fixed-function-abi"="true" "metaxgpu-bsm-direct-address"="true" "metaxgpu-implicitarg-num-bytes"="80" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="128" "metaxgpu-min-blocks"="1" "metaxgpu-new-streg-abi"="false" "metaxgpu-pk-fma"="false" "metaxgpu-resource-usage"="false" "metaxgpu-sched-select"="default" "metaxgpu-use-dim-intrinsic"="false" "no-trapping-math"="true" "prec-div"="false" "prec-sqrt"="false" "scalarize-global-loads"="true" "shfl-combine"="true" "stack-protector-buffer-size"="8" "target-cpu"="xcore1000" "target-features"="+xcore1000" "uniform-work-group-size"="true" }
attributes #6 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { convergent nounwind willreturn }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #10 = { nomerge }

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
!30 = !{!31, !31, i64 0}
!31 = !{!"float", !5, i64 0}
!32 = !{!12, !12, i64 0}
!33 = !{i32 2, i32 -1, i32 -1, i32 -1}
!34 = !{i32 -1, i32 4, i32 -1, i32 -1}
