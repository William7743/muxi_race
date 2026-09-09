; ModuleID = '/data/nsa_codex_20260909_r1/results/ir132_133/nsa132_case6.bc'
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
  %13 = lshr i32 %8, 5
  %14 = icmp ugt i32 %12, %13
  br i1 %14, label %519, label %15

15:                                               ; preds = %5
  %16 = shl nuw nsw i32 %12, 5
  %17 = and i32 %16, 992
  %18 = tail call noundef range(i32 0, 1024) i32 @llvm.mxc.thread.id.x(), !range !26
  %19 = lshr i32 %18, 2
  %20 = and i32 %19, 252
  %21 = add nuw nsw i32 %17, %20
  %22 = icmp ugt i32 %21, %8
  %23 = select i1 %22, float 0xFFF0000000000000, float 0.000000e+00
  %24 = insertelement <4 x float> poison, float %23, i64 0
  %25 = icmp ult i32 %21, %8
  %26 = select i1 %25, float 0.000000e+00, float 0xFFF0000000000000
  %27 = insertelement <4 x float> %24, float %26, i64 1
  %28 = or disjoint i32 %21, 2
  %29 = icmp ugt i32 %28, %8
  %30 = select i1 %29, float 0xFFF0000000000000, float 0.000000e+00
  %31 = insertelement <4 x float> %27, float %30, i64 2
  %32 = or disjoint i32 %21, 3
  %33 = icmp ugt i32 %32, %8
  %34 = select i1 %33, float 0xFFF0000000000000, float 0.000000e+00
  %35 = insertelement <4 x float> %31, float %34, i64 3
  %36 = shl nsw i32 %6, 17
  %37 = shl i32 %12, 12
  %38 = and i32 %37, 126976
  %39 = shl nuw nsw i32 %18, 4
  %40 = and i32 %39, 16256
  %41 = shl nuw nsw i32 %18, 3
  %42 = and i32 %41, 56
  %43 = or disjoint i32 %40, %36
  %44 = or disjoint i32 %43, %42
  %45 = add nuw nsw i32 %44, %38
  %46 = zext nneg i32 %45 to i64
  %47 = getelementptr inbounds %struct.__half, ptr addrspace(4) %4, i64 %46
  %48 = load i32, ptr addrspace(4) %47, align 16, !tbaa !17
  %49 = getelementptr inbounds i8, ptr addrspace(4) %47, i64 4
  %50 = load i32, ptr addrspace(4) %49, align 4, !tbaa !17
  %51 = getelementptr inbounds i8, ptr addrspace(4) %47, i64 8
  %52 = load i32, ptr addrspace(4) %51, align 8, !tbaa !17
  %53 = getelementptr inbounds i8, ptr addrspace(4) %47, i64 12
  %54 = load i32, ptr addrspace(4) %53, align 4, !tbaa !17
  %55 = getelementptr inbounds i8, ptr addrspace(4) %47, i64 4096
  %56 = load i32, ptr addrspace(4) %55, align 16, !tbaa !17
  %57 = getelementptr inbounds i8, ptr addrspace(4) %47, i64 4100
  %58 = load i32, ptr addrspace(4) %57, align 4, !tbaa !17
  %59 = getelementptr inbounds i8, ptr addrspace(4) %47, i64 4104
  %60 = load i32, ptr addrspace(4) %59, align 8, !tbaa !17
  %61 = getelementptr inbounds i8, ptr addrspace(4) %47, i64 4108
  %62 = load i32, ptr addrspace(4) %61, align 4, !tbaa !17
  %63 = shl nsw i32 %6, 21
  %64 = shl nsw i32 %8, 11
  %65 = add nuw nsw i32 %63, %64
  %66 = add nuw nsw i32 %65, %40
  %67 = or disjoint i32 %66, %42
  %68 = and i32 %41, 8128
  %69 = and i32 %41, 32
  %70 = add nuw nsw i32 %69, %18
  %71 = and i32 %70, 32
  %72 = and i32 %41, 16
  %73 = add nuw nsw i32 %72, %18
  %74 = and i32 %73, 16
  %75 = mul nuw nsw i32 %18, 9
  %76 = and i32 %75, 8
  %77 = or disjoint i32 %71, %76
  %78 = or disjoint i32 %77, %68
  %79 = or disjoint i32 %78, %74
  %80 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %79
  %81 = shl nuw nsw i32 %18, 6
  %82 = and i32 %81, 960
  %83 = lshr i32 %18, 1
  %84 = lshr i32 %18, 5
  %85 = add nuw nsw i32 %84, %18
  %86 = shl nuw nsw i32 %85, 3
  %87 = and i32 %86, 8
  %88 = and i32 %19, 4
  %89 = or disjoint i32 %88, %82
  %90 = or disjoint i32 %89, %87
  %91 = and i32 %39, 15360
  %92 = or disjoint i32 %82, %91
  %93 = or disjoint i32 %92, %88
  %94 = or disjoint i32 %93, %87
  %95 = zext nneg i32 %67 to i64
  %96 = getelementptr inbounds %struct.__half, ptr addrspace(4) %3, i64 %95
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %80, ptr addrspace(4) noundef align 16 dereferenceable(16) %96, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %97 = getelementptr inbounds %struct.__half, ptr addrspace(4) %1, i64 %46
  %98 = or disjoint i32 %68, %71
  %99 = or disjoint i32 %98, %74
  %100 = or disjoint i32 %99, %76
  %101 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %100
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %101, ptr addrspace(4) noundef align 16 dereferenceable(16) %97, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  %102 = getelementptr inbounds i8, ptr addrspace(4) %97, i64 4096
  %103 = add nuw nsw i32 %68, 1024
  %104 = or disjoint i32 %103, %71
  %105 = or disjoint i32 %104, %74
  %106 = or disjoint i32 %105, %76
  %107 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %106
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %107, ptr addrspace(4) noundef align 16 dereferenceable(16) %102, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %108 = shl nuw nsw i32 %19, 5
  %109 = and i32 %108, 32
  %110 = shl nuw nsw i32 %83, 4
  %111 = and i32 %110, 16
  %112 = or disjoint i32 %90, %111
  %113 = or disjoint i32 %112, %109
  %114 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %113
  %115 = load <4 x half>, ptr addrspace(3) %114, align 8
  %116 = or disjoint i32 %94, %111
  %117 = or disjoint i32 %116, %109
  %118 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %117
  %119 = load <4 x half>, ptr addrspace(3) %118, align 8
  %120 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %119, <4 x half> %115, <4 x float> %35)
  %121 = xor i32 %111, 16
  %122 = or disjoint i32 %90, %121
  %123 = or disjoint i32 %122, %109
  %124 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %123
  %125 = load <4 x half>, ptr addrspace(3) %124, align 8
  %126 = or disjoint i32 %94, %121
  %127 = or disjoint i32 %126, %109
  %128 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %127
  %129 = load <4 x half>, ptr addrspace(3) %128, align 8
  %130 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %129, <4 x half> %125, <4 x float> %120)
  %131 = add nuw nsw i32 %19, 1
  %132 = shl nuw nsw i32 %131, 5
  %133 = and i32 %132, 32
  %134 = or disjoint i32 %112, %133
  %135 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %134
  %136 = load <4 x half>, ptr addrspace(3) %135, align 8
  %137 = or disjoint i32 %116, %133
  %138 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %137
  %139 = load <4 x half>, ptr addrspace(3) %138, align 8
  %140 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %139, <4 x half> %136, <4 x float> %130)
  %141 = or disjoint i32 %122, %133
  %142 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %141
  %143 = load <4 x half>, ptr addrspace(3) %142, align 8
  %144 = or disjoint i32 %126, %133
  %145 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %144
  %146 = load <4 x half>, ptr addrspace(3) %145, align 8
  %147 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %146, <4 x half> %143, <4 x float> %140)
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %148 = or disjoint i64 %95, 64
  %149 = getelementptr inbounds %struct.__half, ptr addrspace(4) %3, i64 %148
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %80, ptr addrspace(4) noundef align 16 dereferenceable(16) %149, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %150 = or disjoint i64 %46, 64
  %151 = getelementptr inbounds %struct.__half, ptr addrspace(4) %1, i64 %150
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %101, ptr addrspace(4) noundef align 16 dereferenceable(16) %151, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  %152 = getelementptr inbounds i8, ptr addrspace(4) %97, i64 4224
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %107, ptr addrspace(4) noundef align 16 dereferenceable(16) %152, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %153 = load <4 x half>, ptr addrspace(3) %114, align 8
  %154 = load <4 x half>, ptr addrspace(3) %118, align 8
  %155 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %154, <4 x half> %153, <4 x float> %147)
  %156 = load <4 x half>, ptr addrspace(3) %124, align 8
  %157 = load <4 x half>, ptr addrspace(3) %128, align 8
  %158 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %157, <4 x half> %156, <4 x float> %155)
  %159 = load <4 x half>, ptr addrspace(3) %135, align 8
  %160 = load <4 x half>, ptr addrspace(3) %138, align 8
  %161 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %160, <4 x half> %159, <4 x float> %158)
  %162 = load <4 x half>, ptr addrspace(3) %142, align 8
  %163 = load <4 x half>, ptr addrspace(3) %145, align 8
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
  %173 = getelementptr inbounds float, ptr addrspace(3) @buf_dyn_shmem, i32 %18
  store float %172, ptr addrspace(3) %173, align 4, !tbaa !30
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %174 = xor i32 %18, 64
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
  %272 = shl nuw nsw i32 %18, 5
  %273 = and i32 %272, 480
  %274 = lshr i32 %18, 6
  %275 = add nuw nsw i32 %274, %19
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
  %288 = and i32 %39, 768
  %289 = lshr i32 %18, 4
  %290 = add nuw nsw i32 %274, %289
  %291 = shl nuw nsw i32 %290, 5
  %292 = and i32 %291, 32
  %293 = and i32 %18, 7
  %294 = or disjoint i32 %76, %68
  %295 = or disjoint i32 %294, %71
  %296 = or disjoint i32 %295, %74
  %297 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %296
  %298 = lshr i32 %48, 16
  %299 = trunc i32 %48 to i16
  %300 = trunc nuw i32 %298 to i16
  %301 = lshr i32 %50, 16
  %302 = trunc i32 %50 to i16
  %303 = trunc nuw i32 %301 to i16
  %304 = lshr i32 %52, 16
  %305 = trunc i32 %52 to i16
  %306 = trunc nuw i32 %304 to i16
  %307 = lshr i32 %54, 16
  %308 = trunc i32 %54 to i16
  %309 = trunc nuw i32 %307 to i16
  %310 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %100
  store i16 %299, ptr addrspace(3) %310, align 16
  %311 = getelementptr inbounds i8, ptr addrspace(3) %310, i32 2
  store i16 %300, ptr addrspace(3) %311, align 2, !tbaa !17
  %312 = getelementptr inbounds i8, ptr addrspace(3) %310, i32 4
  store i16 %302, ptr addrspace(3) %312, align 4
  %313 = getelementptr inbounds i8, ptr addrspace(3) %310, i32 6
  store i16 %303, ptr addrspace(3) %313, align 2, !tbaa !17
  %314 = getelementptr inbounds i8, ptr addrspace(3) %310, i32 8
  store i16 %305, ptr addrspace(3) %314, align 8
  %315 = getelementptr inbounds i8, ptr addrspace(3) %310, i32 10
  store i16 %306, ptr addrspace(3) %315, align 2, !tbaa !17
  %316 = getelementptr inbounds i8, ptr addrspace(3) %310, i32 12
  store i16 %308, ptr addrspace(3) %316, align 4
  %317 = getelementptr inbounds i8, ptr addrspace(3) %310, i32 14
  store i16 %309, ptr addrspace(3) %317, align 2, !tbaa !17
  %318 = lshr i32 %56, 16
  %319 = trunc i32 %56 to i16
  %320 = trunc nuw i32 %318 to i16
  %321 = lshr i32 %58, 16
  %322 = trunc i32 %58 to i16
  %323 = trunc nuw i32 %321 to i16
  %324 = lshr i32 %60, 16
  %325 = trunc i32 %60 to i16
  %326 = trunc nuw i32 %324 to i16
  %327 = lshr i32 %62, 16
  %328 = trunc i32 %62 to i16
  %329 = trunc nuw i32 %327 to i16
  %330 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %106
  store i16 %319, ptr addrspace(3) %330, align 16
  %331 = getelementptr inbounds i8, ptr addrspace(3) %330, i32 2
  store i16 %320, ptr addrspace(3) %331, align 2, !tbaa !17
  %332 = getelementptr inbounds i8, ptr addrspace(3) %330, i32 4
  store i16 %322, ptr addrspace(3) %332, align 4
  %333 = getelementptr inbounds i8, ptr addrspace(3) %330, i32 6
  store i16 %323, ptr addrspace(3) %333, align 2, !tbaa !17
  %334 = getelementptr inbounds i8, ptr addrspace(3) %330, i32 8
  store i16 %325, ptr addrspace(3) %334, align 8
  %335 = getelementptr inbounds i8, ptr addrspace(3) %330, i32 10
  store i16 %326, ptr addrspace(3) %335, align 2, !tbaa !17
  %336 = getelementptr inbounds i8, ptr addrspace(3) %330, i32 12
  store i16 %328, ptr addrspace(3) %336, align 4
  %337 = getelementptr inbounds i8, ptr addrspace(3) %330, i32 14
  store i16 %329, ptr addrspace(3) %337, align 2, !tbaa !17
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %338 = shl nuw nsw i32 %19, 4
  %339 = and i32 %338, 16
  %340 = or disjoint i32 %273, %339
  %341 = or disjoint i32 %340, %88
  %342 = or disjoint i32 %341, %280
  %343 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %342
  %344 = load <4 x half>, ptr addrspace(3) %343, align 8
  %345 = or disjoint i32 %288, %292
  %346 = and i32 %18, 8
  %347 = or disjoint i32 %345, %346
  %348 = or disjoint i32 %347, %293
  %349 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %348
  %350 = load half, ptr addrspace(3) %349, align 2, !tbaa !32
  %351 = insertelement <4 x half> poison, half %350, i64 0
  %352 = or disjoint i32 %346, %345
  %353 = or disjoint i32 %352, %293
  %354 = xor i32 %353, 8
  %355 = or disjoint i32 %354, 64
  %356 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %355
  %357 = load half, ptr addrspace(3) %356, align 2, !tbaa !32
  %358 = insertelement <4 x half> %351, half %357, i64 1
  %359 = or disjoint i32 %348, 144
  %360 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %359
  %361 = load half, ptr addrspace(3) %360, align 2, !tbaa !32
  %362 = insertelement <4 x half> %358, half %361, i64 2
  %363 = or disjoint i32 %354, 208
  %364 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %363
  %365 = load half, ptr addrspace(3) %364, align 2, !tbaa !32
  %366 = insertelement <4 x half> %362, half %365, i64 3
  %367 = or disjoint i32 %348, 16
  %368 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %367
  %369 = load half, ptr addrspace(3) %368, align 2, !tbaa !32
  %370 = insertelement <4 x half> poison, half %369, i64 0
  %371 = or disjoint i32 %354, 80
  %372 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %371
  %373 = load half, ptr addrspace(3) %372, align 2, !tbaa !32
  %374 = insertelement <4 x half> %370, half %373, i64 1
  %375 = or disjoint i32 %348, 128
  %376 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %375
  %377 = load half, ptr addrspace(3) %376, align 2, !tbaa !32
  %378 = insertelement <4 x half> %374, half %377, i64 2
  %379 = or disjoint i32 %354, 192
  %380 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %379
  %381 = load half, ptr addrspace(3) %380, align 2, !tbaa !32
  %382 = insertelement <4 x half> %378, half %381, i64 3
  %383 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %366, <4 x half> %344, <4 x float> zeroinitializer)
  %384 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %382, <4 x half> %344, <4 x float> zeroinitializer)
  %385 = shl nuw nsw i32 %131, 4
  %386 = and i32 %385, 16
  %387 = or disjoint i32 %273, %386
  %388 = or disjoint i32 %387, %88
  %389 = or disjoint i32 %388, %280
  %390 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %389
  %391 = load <4 x half>, ptr addrspace(3) %390, align 8
  %392 = or disjoint i32 %348, 1024
  %393 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %392
  %394 = load half, ptr addrspace(3) %393, align 2, !tbaa !32
  %395 = insertelement <4 x half> poison, half %394, i64 0
  %396 = or disjoint i32 %354, 1088
  %397 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %396
  %398 = load half, ptr addrspace(3) %397, align 2, !tbaa !32
  %399 = insertelement <4 x half> %395, half %398, i64 1
  %400 = or disjoint i32 %348, 1168
  %401 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %400
  %402 = load half, ptr addrspace(3) %401, align 2, !tbaa !32
  %403 = insertelement <4 x half> %399, half %402, i64 2
  %404 = or disjoint i32 %354, 1232
  %405 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %404
  %406 = load half, ptr addrspace(3) %405, align 2, !tbaa !32
  %407 = insertelement <4 x half> %403, half %406, i64 3
  %408 = or disjoint i32 %348, 1040
  %409 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %408
  %410 = load half, ptr addrspace(3) %409, align 2, !tbaa !32
  %411 = insertelement <4 x half> poison, half %410, i64 0
  %412 = or disjoint i32 %354, 1104
  %413 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %412
  %414 = load half, ptr addrspace(3) %413, align 2, !tbaa !32
  %415 = insertelement <4 x half> %411, half %414, i64 1
  %416 = or disjoint i32 %348, 1152
  %417 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %416
  %418 = load half, ptr addrspace(3) %417, align 2, !tbaa !32
  %419 = insertelement <4 x half> %415, half %418, i64 2
  %420 = or disjoint i32 %354, 1216
  %421 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %420
  %422 = load half, ptr addrspace(3) %421, align 2, !tbaa !32
  %423 = insertelement <4 x half> %419, half %422, i64 3
  %424 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %407, <4 x half> %391, <4 x float> %383)
  %425 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %423, <4 x half> %391, <4 x float> %384)
  %426 = or disjoint i32 %44, 64
  %427 = add nuw nsw i32 %426, %38
  %428 = zext nneg i32 %427 to i64
  %429 = shl nuw nsw i32 %275, 5
  %430 = and i32 %429, 32
  %431 = or disjoint i32 %82, %430
  %432 = or disjoint i32 %431, %87
  %433 = extractelement <4 x float> %425, i64 3
  %434 = extractelement <4 x float> %425, i64 2
  %435 = extractelement <4 x float> %425, i64 1
  %436 = extractelement <4 x float> %425, i64 0
  %437 = extractelement <4 x float> %424, i64 3
  %438 = extractelement <4 x float> %424, i64 2
  %439 = extractelement <4 x float> %424, i64 1
  %440 = extractelement <4 x float> %424, i64 0
  %441 = fptrunc float %440 to half
  %442 = fptrunc float %439 to half
  %443 = fptrunc float %438 to half
  %444 = fptrunc float %437 to half
  %445 = fptrunc float %436 to half
  %446 = fptrunc float %435 to half
  %447 = fptrunc float %434 to half
  %448 = fptrunc float %433 to half
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  %449 = or disjoint i32 %432, %111
  %450 = or disjoint i32 %449, %88
  %451 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %450
  store half %441, ptr addrspace(3) %451, align 8
  %452 = getelementptr inbounds i8, ptr addrspace(3) %451, i32 2
  store half %442, ptr addrspace(3) %452, align 2
  %453 = getelementptr inbounds i8, ptr addrspace(3) %451, i32 4
  store half %443, ptr addrspace(3) %453, align 4
  %454 = getelementptr inbounds i8, ptr addrspace(3) %451, i32 6
  store half %444, ptr addrspace(3) %454, align 2
  %455 = or disjoint i32 %432, %121
  %456 = or disjoint i32 %455, %88
  %457 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %456
  store half %445, ptr addrspace(3) %457, align 8
  %458 = getelementptr inbounds i8, ptr addrspace(3) %457, i32 2
  store half %446, ptr addrspace(3) %458, align 2
  %459 = getelementptr inbounds i8, ptr addrspace(3) %457, i32 4
  store half %447, ptr addrspace(3) %459, align 4
  %460 = getelementptr inbounds i8, ptr addrspace(3) %457, i32 6
  store half %448, ptr addrspace(3) %460, align 2
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %461 = getelementptr inbounds %struct.__half, ptr addrspace(1) %2, i64 %95
  tail call void @llvm.memcpy.p1.p3.i64(ptr addrspace(1) noundef align 16 dereferenceable(16) %461, ptr addrspace(3) noundef align 16 dereferenceable(16) %297, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !33
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  %462 = getelementptr inbounds %struct.__half, ptr addrspace(4) %4, i64 %428
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %310, ptr addrspace(4) noundef align 16 dereferenceable(16) %462, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !34
  %463 = getelementptr inbounds i8, ptr addrspace(4) %462, i64 4096
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %330, ptr addrspace(4) noundef align 16 dereferenceable(16) %463, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !34
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %464 = load <4 x half>, ptr addrspace(3) %343, align 8
  %465 = load half, ptr addrspace(3) %349, align 2, !tbaa !32
  %466 = insertelement <4 x half> poison, half %465, i64 0
  %467 = load half, ptr addrspace(3) %356, align 2, !tbaa !32
  %468 = insertelement <4 x half> %466, half %467, i64 1
  %469 = load half, ptr addrspace(3) %360, align 2, !tbaa !32
  %470 = insertelement <4 x half> %468, half %469, i64 2
  %471 = load half, ptr addrspace(3) %364, align 2, !tbaa !32
  %472 = insertelement <4 x half> %470, half %471, i64 3
  %473 = load half, ptr addrspace(3) %368, align 2, !tbaa !32
  %474 = insertelement <4 x half> poison, half %473, i64 0
  %475 = load half, ptr addrspace(3) %372, align 2, !tbaa !32
  %476 = insertelement <4 x half> %474, half %475, i64 1
  %477 = load half, ptr addrspace(3) %376, align 2, !tbaa !32
  %478 = insertelement <4 x half> %476, half %477, i64 2
  %479 = load half, ptr addrspace(3) %380, align 2, !tbaa !32
  %480 = insertelement <4 x half> %478, half %479, i64 3
  %481 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %472, <4 x half> %464, <4 x float> zeroinitializer)
  %482 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %480, <4 x half> %464, <4 x float> zeroinitializer)
  %483 = load <4 x half>, ptr addrspace(3) %390, align 8
  %484 = load half, ptr addrspace(3) %393, align 2, !tbaa !32
  %485 = insertelement <4 x half> poison, half %484, i64 0
  %486 = load half, ptr addrspace(3) %397, align 2, !tbaa !32
  %487 = insertelement <4 x half> %485, half %486, i64 1
  %488 = load half, ptr addrspace(3) %401, align 2, !tbaa !32
  %489 = insertelement <4 x half> %487, half %488, i64 2
  %490 = load half, ptr addrspace(3) %405, align 2, !tbaa !32
  %491 = insertelement <4 x half> %489, half %490, i64 3
  %492 = load half, ptr addrspace(3) %409, align 2, !tbaa !32
  %493 = insertelement <4 x half> poison, half %492, i64 0
  %494 = load half, ptr addrspace(3) %413, align 2, !tbaa !32
  %495 = insertelement <4 x half> %493, half %494, i64 1
  %496 = load half, ptr addrspace(3) %417, align 2, !tbaa !32
  %497 = insertelement <4 x half> %495, half %496, i64 2
  %498 = load half, ptr addrspace(3) %421, align 2, !tbaa !32
  %499 = insertelement <4 x half> %497, half %498, i64 3
  %500 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %491, <4 x half> %483, <4 x float> %481)
  %501 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %499, <4 x half> %483, <4 x float> %482)
  %502 = extractelement <4 x float> %501, i64 3
  %503 = extractelement <4 x float> %501, i64 2
  %504 = extractelement <4 x float> %501, i64 1
  %505 = extractelement <4 x float> %501, i64 0
  %506 = extractelement <4 x float> %500, i64 3
  %507 = extractelement <4 x float> %500, i64 2
  %508 = extractelement <4 x float> %500, i64 1
  %509 = extractelement <4 x float> %500, i64 0
  %510 = fptrunc float %509 to half
  %511 = fptrunc float %508 to half
  %512 = fptrunc float %507 to half
  %513 = fptrunc float %506 to half
  %514 = fptrunc float %505 to half
  %515 = fptrunc float %504 to half
  %516 = fptrunc float %503 to half
  %517 = fptrunc float %502 to half
  fence syncscope("warp") release
  tail call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  store half %510, ptr addrspace(3) %451, align 8
  store half %511, ptr addrspace(3) %452, align 2
  store half %512, ptr addrspace(3) %453, align 4
  store half %513, ptr addrspace(3) %454, align 2
  store half %514, ptr addrspace(3) %457, align 8
  store half %515, ptr addrspace(3) %458, align 2
  store half %516, ptr addrspace(3) %459, align 4
  store half %517, ptr addrspace(3) %460, align 2
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %518 = getelementptr inbounds %struct.__half, ptr addrspace(1) %2, i64 %148
  tail call void @llvm.memcpy.p1.p3.i64(ptr addrspace(1) noundef align 16 dereferenceable(16) %518, ptr addrspace(3) noundef align 16 dereferenceable(16) %297, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !33
  br label %519

519:                                              ; preds = %15, %5
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
