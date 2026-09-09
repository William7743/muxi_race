; ModuleID = '/data/nsa_codex_20260909_r1/results/ir134_136/nsa136_case11.bc'
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
define protected metaxgpu_kernel void @gather_kernel_kernel(ptr addrspace(1) noalias nocapture noundef readonly %0, ptr addrspace(4) noalias nocapture noundef readonly %1, ptr addrspace(1) noalias nocapture noundef writeonly %2, ptr addrspace(4) noalias nocapture noundef readonly %3, ptr addrspace(4) noalias nocapture noundef readonly %4) local_unnamed_addr #5 {
  %6 = tail call noundef range(i32 0, 2147483647) i32 @llvm.mxc.block.id.y(), !range !16
  %7 = shl nsw i32 %6, 11
  %8 = tail call noundef range(i32 0, 2147483647) i32 @llvm.mxc.block.id.x(), !range !16
  %9 = shl nsw i32 %8, 2
  %10 = add nuw nsw i32 %7, %9
  %11 = tail call noundef range(i32 0, 1024) i32 @llvm.mxc.thread.id.x(), !range !26
  %12 = lshr i32 %11, 5
  %13 = and i32 %12, 30
  %14 = add nuw nsw i32 %10, %13
  %15 = lshr i32 %11, 2
  %16 = and i32 %15, 12
  %17 = tail call i32 @llvm.smin.i32(i32 %8, i32 511)
  %18 = zext nneg i32 %14 to i64
  %19 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %18
  %20 = load i32, ptr addrspace(1) %19, align 4, !tbaa !17
  %21 = shl nsw i32 %20, 4
  %22 = or disjoint i32 %21, %16
  %23 = icmp slt i32 %20, 0
  %24 = icmp sgt i32 %22, %17
  %25 = select i1 %23, i1 true, i1 %24
  %26 = select i1 %25, float 0xFFF0000000000000, float 0.000000e+00
  %27 = insertelement <4 x float> poison, float %26, i64 0
  %28 = icmp sge i32 %22, %17
  %29 = select i1 %23, i1 true, i1 %28
  %30 = select i1 %29, float 0xFFF0000000000000, float 0.000000e+00
  %31 = insertelement <4 x float> %27, float %30, i64 1
  %32 = or disjoint i32 %22, 2
  %33 = icmp sgt i32 %32, %17
  %34 = select i1 %23, i1 true, i1 %33
  %35 = select i1 %34, float 0xFFF0000000000000, float 0.000000e+00
  %36 = insertelement <4 x float> %31, float %35, i64 2
  %37 = or disjoint i32 %22, 3
  %38 = icmp sgt i32 %37, %17
  %39 = select i1 %23, i1 true, i1 %38
  %40 = select i1 %39, float 0xFFF0000000000000, float 0.000000e+00
  %41 = insertelement <4 x float> %36, float %40, i64 3
  %42 = or disjoint i64 %18, 1
  %43 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %42
  %44 = load i32, ptr addrspace(1) %43, align 4, !tbaa !17
  %45 = shl nsw i32 %44, 4
  %46 = or disjoint i32 %45, %16
  %47 = icmp slt i32 %44, 0
  %48 = icmp sgt i32 %46, %17
  %49 = select i1 %47, i1 true, i1 %48
  %50 = select i1 %49, float 0xFFF0000000000000, float 0.000000e+00
  %51 = insertelement <4 x float> poison, float %50, i64 0
  %52 = icmp sge i32 %46, %17
  %53 = select i1 %47, i1 true, i1 %52
  %54 = select i1 %53, float 0xFFF0000000000000, float 0.000000e+00
  %55 = insertelement <4 x float> %51, float %54, i64 1
  %56 = or disjoint i32 %46, 2
  %57 = icmp sgt i32 %56, %17
  %58 = select i1 %47, i1 true, i1 %57
  %59 = select i1 %58, float 0xFFF0000000000000, float 0.000000e+00
  %60 = insertelement <4 x float> %55, float %59, i64 2
  %61 = or disjoint i32 %46, 3
  %62 = icmp sgt i32 %61, %17
  %63 = select i1 %47, i1 true, i1 %62
  %64 = select i1 %63, float 0xFFF0000000000000, float 0.000000e+00
  %65 = insertelement <4 x float> %60, float %64, i64 3
  %66 = shl nsw i32 %6, 19
  %67 = shl nsw i32 %8, 10
  %68 = add nuw nsw i32 %66, %67
  %69 = shl nuw nsw i32 %11, 3
  %70 = and i32 %69, 8128
  %71 = add nuw nsw i32 %68, %70
  %72 = shl nuw nsw i32 %11, 2
  %73 = and i32 %72, 28
  %74 = or disjoint i32 %71, %73
  %75 = add nuw nsw i32 %12, %15
  %76 = shl nuw nsw i32 %75, 4
  %77 = and i32 %76, 16
  %78 = lshr i32 %11, 4
  %79 = lshr i32 %11, 1
  %80 = add nuw nsw i32 %78, %79
  %81 = shl nuw nsw i32 %80, 3
  %82 = and i32 %81, 8
  %83 = and i32 %72, 4068
  %84 = or disjoint i32 %83, %77
  %85 = or disjoint i32 %84, %82
  %86 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %85
  %87 = lshr i32 %11, 6
  %88 = add nuw nsw i32 %10, %87
  %89 = and i32 %15, 15
  %90 = and i32 %69, 8160
  %91 = and i32 %69, 16
  %92 = add nuw nsw i32 %91, %11
  %93 = and i32 %92, 16
  %94 = mul nuw nsw i32 %11, 9
  %95 = and i32 %94, 8
  %96 = zext nneg i32 %6 to i64
  %97 = shl nuw nsw i64 %96, 15
  %98 = and i32 %69, 24
  %99 = zext nneg i32 %98 to i64
  %100 = shl nuw nsw i32 %11, 5
  %101 = and i32 %100, 480
  %102 = add nuw nsw i32 %12, %79
  %103 = shl nuw nsw i32 %102, 3
  %104 = and i32 %103, 8
  %105 = and i32 %15, 4
  %106 = or disjoint i32 %105, %101
  %107 = or disjoint i32 %106, %104
  %108 = shl nuw nsw i32 %11, 4
  %109 = and i32 %108, 15360
  %110 = zext nneg i32 %88 to i64
  %111 = zext nneg i32 %74 to i64
  %112 = getelementptr inbounds %struct.__half, ptr addrspace(4) %3, i64 %111
  %113 = load i64, ptr addrspace(4) %112, align 8
  store i64 %113, ptr addrspace(3) %86, align 8
  %114 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %110
  %115 = load i32, ptr addrspace(1) %114, align 4, !tbaa !17
  %116 = shl nsw i32 %115, 4
  %117 = or disjoint i32 %116, %89
  %118 = icmp slt i32 %115, 0
  %119 = icmp sgt i32 %117, %17
  %120 = select i1 %118, i1 true, i1 %119
  br i1 %120, label %134, label %121

121:                                              ; preds = %5
  %122 = zext nneg i32 %117 to i64
  %123 = shl nuw nsw i64 %122, 6
  %124 = add nuw nsw i64 %123, %97
  %125 = or disjoint i64 %124, %99
  %126 = getelementptr inbounds %struct.__half, ptr addrspace(4) %1, i64 %125
  %127 = load i32, ptr addrspace(4) %126, align 16, !tbaa !17
  %128 = getelementptr inbounds i8, ptr addrspace(4) %126, i64 4
  %129 = load i32, ptr addrspace(4) %128, align 4, !tbaa !17
  %130 = getelementptr inbounds i8, ptr addrspace(4) %126, i64 8
  %131 = load i32, ptr addrspace(4) %130, align 8, !tbaa !17
  %132 = getelementptr inbounds i8, ptr addrspace(4) %126, i64 12
  %133 = load i32, ptr addrspace(4) %132, align 4, !tbaa !17
  br label %134

134:                                              ; preds = %121, %5
  %135 = phi i32 [ %127, %121 ], [ 0, %5 ]
  %136 = phi i32 [ %129, %121 ], [ 0, %5 ]
  %137 = phi i32 [ %131, %121 ], [ 0, %5 ]
  %138 = phi i32 [ %133, %121 ], [ 0, %5 ]
  %139 = or disjoint i32 %90, %93
  %140 = or disjoint i32 %139, %95
  %141 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %140
  store i32 %135, ptr addrspace(3) %141, align 16, !tbaa !17
  %142 = getelementptr inbounds i8, ptr addrspace(3) %141, i32 4
  store i32 %136, ptr addrspace(3) %142, align 4, !tbaa !17
  %143 = getelementptr inbounds i8, ptr addrspace(3) %141, i32 8
  store i32 %137, ptr addrspace(3) %143, align 8, !tbaa !17
  %144 = getelementptr inbounds i8, ptr addrspace(3) %141, i32 12
  store i32 %138, ptr addrspace(3) %144, align 4, !tbaa !17
  %145 = getelementptr inbounds i8, ptr addrspace(1) %114, i64 8
  %146 = load i32, ptr addrspace(1) %145, align 4, !tbaa !17
  %147 = shl nsw i32 %146, 4
  %148 = or disjoint i32 %147, %89
  %149 = icmp slt i32 %146, 0
  %150 = icmp sgt i32 %148, %17
  %151 = select i1 %149, i1 true, i1 %150
  br i1 %151, label %165, label %152

152:                                              ; preds = %134
  %153 = zext nneg i32 %148 to i64
  %154 = shl nuw nsw i64 %153, 6
  %155 = add nuw nsw i64 %154, %97
  %156 = or disjoint i64 %155, %99
  %157 = getelementptr inbounds %struct.__half, ptr addrspace(4) %1, i64 %156
  %158 = load i32, ptr addrspace(4) %157, align 16, !tbaa !17
  %159 = getelementptr inbounds i8, ptr addrspace(4) %157, i64 4
  %160 = load i32, ptr addrspace(4) %159, align 4, !tbaa !17
  %161 = getelementptr inbounds i8, ptr addrspace(4) %157, i64 8
  %162 = load i32, ptr addrspace(4) %161, align 8, !tbaa !17
  %163 = getelementptr inbounds i8, ptr addrspace(4) %157, i64 12
  %164 = load i32, ptr addrspace(4) %163, align 4, !tbaa !17
  br label %165

165:                                              ; preds = %152, %134
  %166 = phi i32 [ %158, %152 ], [ 0, %134 ]
  %167 = phi i32 [ %160, %152 ], [ 0, %134 ]
  %168 = phi i32 [ %162, %152 ], [ 0, %134 ]
  %169 = phi i32 [ %164, %152 ], [ 0, %134 ]
  %170 = add nuw nsw i32 %90, 1024
  %171 = or disjoint i32 %170, %93
  %172 = or disjoint i32 %171, %95
  %173 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %172
  store i32 %166, ptr addrspace(3) %173, align 16, !tbaa !17
  %174 = getelementptr inbounds i8, ptr addrspace(3) %173, i32 4
  store i32 %167, ptr addrspace(3) %174, align 4, !tbaa !17
  %175 = getelementptr inbounds i8, ptr addrspace(3) %173, i32 8
  store i32 %168, ptr addrspace(3) %175, align 8, !tbaa !17
  %176 = getelementptr inbounds i8, ptr addrspace(3) %173, i32 12
  store i32 %169, ptr addrspace(3) %176, align 4, !tbaa !17
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %177 = shl nuw nsw i32 %15, 4
  %178 = and i32 %177, 16
  %179 = or disjoint i32 %107, %178
  %180 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %179
  %181 = load <4 x half>, ptr addrspace(3) %180, align 8
  %182 = or disjoint i32 %109, %101
  %183 = or disjoint i32 %182, %178
  %184 = or disjoint i32 %183, %104
  %185 = or disjoint i32 %184, %105
  %186 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %185
  %187 = load <4 x half>, ptr addrspace(3) %186, align 8
  %188 = or disjoint i32 %182, 512
  %189 = or disjoint i32 %188, %178
  %190 = or disjoint i32 %189, %104
  %191 = or disjoint i32 %190, %105
  %192 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %191
  %193 = load <4 x half>, ptr addrspace(3) %192, align 8
  %194 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %187, <4 x half> %181, <4 x float> %41)
  %195 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %193, <4 x half> %181, <4 x float> %65)
  %196 = add nuw nsw i32 %15, 1
  %197 = shl nuw nsw i32 %196, 4
  %198 = and i32 %197, 16
  %199 = or disjoint i32 %107, %198
  %200 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %199
  %201 = load <4 x half>, ptr addrspace(3) %200, align 8
  %202 = or disjoint i32 %182, %198
  %203 = or disjoint i32 %202, %104
  %204 = or disjoint i32 %203, %105
  %205 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %204
  %206 = load <4 x half>, ptr addrspace(3) %205, align 8
  %207 = or disjoint i32 %188, %198
  %208 = or disjoint i32 %207, %104
  %209 = or disjoint i32 %208, %105
  %210 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %209
  %211 = load <4 x half>, ptr addrspace(3) %210, align 8
  %212 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %206, <4 x half> %201, <4 x float> %194)
  %213 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %211, <4 x half> %201, <4 x float> %195)
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %214 = or disjoint i64 %111, 32
  %215 = getelementptr inbounds %struct.__half, ptr addrspace(4) %3, i64 %214
  %216 = load i64, ptr addrspace(4) %215, align 8
  store i64 %216, ptr addrspace(3) %86, align 8
  br i1 %120, label %231, label %217

217:                                              ; preds = %165
  %218 = zext nneg i32 %117 to i64
  %219 = shl nuw nsw i64 %218, 6
  %220 = add nuw nsw i64 %219, %97
  %221 = or disjoint i64 %220, %99
  %222 = or disjoint i64 %221, 32
  %223 = getelementptr inbounds %struct.__half, ptr addrspace(4) %1, i64 %222
  %224 = load i32, ptr addrspace(4) %223, align 16, !tbaa !17
  %225 = getelementptr inbounds i8, ptr addrspace(4) %223, i64 4
  %226 = load i32, ptr addrspace(4) %225, align 4, !tbaa !17
  %227 = getelementptr inbounds i8, ptr addrspace(4) %223, i64 8
  %228 = load i32, ptr addrspace(4) %227, align 8, !tbaa !17
  %229 = getelementptr inbounds i8, ptr addrspace(4) %223, i64 12
  %230 = load i32, ptr addrspace(4) %229, align 4, !tbaa !17
  br label %231

231:                                              ; preds = %217, %165
  %232 = phi i32 [ %224, %217 ], [ 0, %165 ]
  %233 = phi i32 [ %226, %217 ], [ 0, %165 ]
  %234 = phi i32 [ %228, %217 ], [ 0, %165 ]
  %235 = phi i32 [ %230, %217 ], [ 0, %165 ]
  store i32 %232, ptr addrspace(3) %141, align 16, !tbaa !17
  store i32 %233, ptr addrspace(3) %142, align 4, !tbaa !17
  store i32 %234, ptr addrspace(3) %143, align 8, !tbaa !17
  store i32 %235, ptr addrspace(3) %144, align 4, !tbaa !17
  br i1 %151, label %250, label %236

236:                                              ; preds = %231
  %237 = zext nneg i32 %148 to i64
  %238 = shl nuw nsw i64 %237, 6
  %239 = add nuw nsw i64 %238, %97
  %240 = or disjoint i64 %239, %99
  %241 = or disjoint i64 %240, 32
  %242 = getelementptr inbounds %struct.__half, ptr addrspace(4) %1, i64 %241
  %243 = load i32, ptr addrspace(4) %242, align 16, !tbaa !17
  %244 = getelementptr inbounds i8, ptr addrspace(4) %242, i64 4
  %245 = load i32, ptr addrspace(4) %244, align 4, !tbaa !17
  %246 = getelementptr inbounds i8, ptr addrspace(4) %242, i64 8
  %247 = load i32, ptr addrspace(4) %246, align 8, !tbaa !17
  %248 = getelementptr inbounds i8, ptr addrspace(4) %242, i64 12
  %249 = load i32, ptr addrspace(4) %248, align 4, !tbaa !17
  br label %250

250:                                              ; preds = %236, %231
  %251 = phi i32 [ %243, %236 ], [ 0, %231 ]
  %252 = phi i32 [ %245, %236 ], [ 0, %231 ]
  %253 = phi i32 [ %247, %236 ], [ 0, %231 ]
  %254 = phi i32 [ %249, %236 ], [ 0, %231 ]
  store i32 %251, ptr addrspace(3) %173, align 16, !tbaa !17
  store i32 %252, ptr addrspace(3) %174, align 4, !tbaa !17
  store i32 %253, ptr addrspace(3) %175, align 8, !tbaa !17
  store i32 %254, ptr addrspace(3) %176, align 4, !tbaa !17
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %255 = load <4 x half>, ptr addrspace(3) %180, align 8
  %256 = load <4 x half>, ptr addrspace(3) %186, align 8
  %257 = load <4 x half>, ptr addrspace(3) %192, align 8
  %258 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %256, <4 x half> %255, <4 x float> %212)
  %259 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %257, <4 x half> %255, <4 x float> %213)
  %260 = load <4 x half>, ptr addrspace(3) %200, align 8
  %261 = load <4 x half>, ptr addrspace(3) %205, align 8
  %262 = load <4 x half>, ptr addrspace(3) %210, align 8
  %263 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %261, <4 x half> %260, <4 x float> %258)
  %264 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %262, <4 x half> %260, <4 x float> %259)
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %265 = extractelement <4 x float> %263, i64 0
  %266 = tail call contract noundef float @llvm.maxnum.f32(float %265, float 0xFFF0000000000000)
  %267 = extractelement <4 x float> %264, i64 0
  %268 = tail call contract noundef float @llvm.maxnum.f32(float %266, float %267)
  %269 = extractelement <4 x float> %263, i64 1
  %270 = tail call contract noundef float @llvm.maxnum.f32(float %268, float %269)
  %271 = extractelement <4 x float> %264, i64 1
  %272 = tail call contract noundef float @llvm.maxnum.f32(float %270, float %271)
  %273 = extractelement <4 x float> %263, i64 2
  %274 = tail call contract noundef float @llvm.maxnum.f32(float %272, float %273)
  %275 = extractelement <4 x float> %264, i64 2
  %276 = tail call contract noundef float @llvm.maxnum.f32(float %274, float %275)
  %277 = extractelement <4 x float> %263, i64 3
  %278 = tail call contract noundef float @llvm.maxnum.f32(float %276, float %277)
  %279 = extractelement <4 x float> %264, i64 3
  %280 = tail call contract noundef float @llvm.maxnum.f32(float %278, float %279)
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %281 = getelementptr inbounds float, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %11
  store float %280, ptr addrspace(3) %281, align 4, !tbaa !27
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %282 = xor i32 %11, 64
  %283 = getelementptr inbounds float, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %282
  %284 = load float, ptr addrspace(3) %283, align 4, !tbaa !27
  %285 = tail call contract noundef float @llvm.maxnum.f32(float %280, float %284)
  %286 = bitcast float %285 to i32
  %287 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #10
  %288 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %287) #10
  %289 = xor i32 %288, 32
  %290 = and i32 %288, -64
  %291 = add nsw i32 %290, 64
  %292 = icmp slt i32 %289, %291
  %293 = select i1 %292, i32 %289, i32 %288
  %294 = shl i32 %293, 2
  %295 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %294, i32 %286)
  %296 = bitcast i32 %295 to float
  %297 = tail call contract noundef float @llvm.maxnum.f32(float %285, float %296)
  %298 = bitcast float %297 to i32
  %299 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #10
  %300 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %299) #10
  %301 = xor i32 %300, 16
  %302 = and i32 %300, -64
  %303 = add nsw i32 %302, 64
  %304 = icmp slt i32 %301, %303
  %305 = select i1 %304, i32 %301, i32 %300
  %306 = shl i32 %305, 2
  %307 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %306, i32 %298)
  %308 = bitcast i32 %307 to float
  %309 = tail call contract noundef float @llvm.maxnum.f32(float %297, float %308)
  %310 = fsub contract float %265, %309
  %311 = fmul contract float %310, 0x3FC7154760000000
  %312 = fcmp contract olt float %311, -1.260000e+02
  %313 = select contract i1 %312, float 6.400000e+01, float 0.000000e+00
  %314 = fadd contract float %311, %313
  %315 = tail call contract float @llvm.exp2.f32(float %314)
  %316 = select contract i1 %312, float 0x3BF0000000000000, float 1.000000e+00
  %317 = fmul contract float %316, %315
  %318 = fsub contract float %269, %309
  %319 = fmul contract float %318, 0x3FC7154760000000
  %320 = fcmp contract olt float %319, -1.260000e+02
  %321 = select contract i1 %320, float 6.400000e+01, float 0.000000e+00
  %322 = fadd contract float %319, %321
  %323 = tail call contract float @llvm.exp2.f32(float %322)
  %324 = select contract i1 %320, float 0x3BF0000000000000, float 1.000000e+00
  %325 = fmul contract float %324, %323
  %326 = fsub contract float %273, %309
  %327 = fmul contract float %326, 0x3FC7154760000000
  %328 = fcmp contract olt float %327, -1.260000e+02
  %329 = select contract i1 %328, float 6.400000e+01, float 0.000000e+00
  %330 = fadd contract float %327, %329
  %331 = tail call contract float @llvm.exp2.f32(float %330)
  %332 = select contract i1 %328, float 0x3BF0000000000000, float 1.000000e+00
  %333 = fmul contract float %332, %331
  %334 = fsub contract float %277, %309
  %335 = fmul contract float %334, 0x3FC7154760000000
  %336 = fcmp contract olt float %335, -1.260000e+02
  %337 = select contract i1 %336, float 6.400000e+01, float 0.000000e+00
  %338 = fadd contract float %335, %337
  %339 = tail call contract float @llvm.exp2.f32(float %338)
  %340 = select contract i1 %336, float 0x3BF0000000000000, float 1.000000e+00
  %341 = fmul contract float %340, %339
  %342 = fsub contract float %267, %309
  %343 = fmul contract float %342, 0x3FC7154760000000
  %344 = fcmp contract olt float %343, -1.260000e+02
  %345 = select contract i1 %344, float 6.400000e+01, float 0.000000e+00
  %346 = fadd contract float %343, %345
  %347 = tail call contract float @llvm.exp2.f32(float %346)
  %348 = select contract i1 %344, float 0x3BF0000000000000, float 1.000000e+00
  %349 = fmul contract float %348, %347
  %350 = fsub contract float %271, %309
  %351 = fmul contract float %350, 0x3FC7154760000000
  %352 = fcmp contract olt float %351, -1.260000e+02
  %353 = select contract i1 %352, float 6.400000e+01, float 0.000000e+00
  %354 = fadd contract float %351, %353
  %355 = tail call contract float @llvm.exp2.f32(float %354)
  %356 = select contract i1 %352, float 0x3BF0000000000000, float 1.000000e+00
  %357 = fmul contract float %356, %355
  %358 = fsub contract float %275, %309
  %359 = fmul contract float %358, 0x3FC7154760000000
  %360 = fcmp contract olt float %359, -1.260000e+02
  %361 = select contract i1 %360, float 6.400000e+01, float 0.000000e+00
  %362 = fadd contract float %359, %361
  %363 = tail call contract float @llvm.exp2.f32(float %362)
  %364 = select contract i1 %360, float 0x3BF0000000000000, float 1.000000e+00
  %365 = fmul contract float %364, %363
  %366 = fsub contract float %279, %309
  %367 = fmul contract float %366, 0x3FC7154760000000
  %368 = fcmp contract olt float %367, -1.260000e+02
  %369 = select contract i1 %368, float 6.400000e+01, float 0.000000e+00
  %370 = fadd contract float %367, %369
  %371 = tail call contract float @llvm.exp2.f32(float %370)
  %372 = select contract i1 %368, float 0x3BF0000000000000, float 1.000000e+00
  %373 = fmul contract float %372, %371
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %374 = fadd contract float %317, 0.000000e+00
  %375 = fadd contract float %374, %349
  %376 = fadd contract float %375, %325
  %377 = fadd contract float %376, %357
  %378 = fadd contract float %377, %333
  %379 = fadd contract float %378, %365
  %380 = fadd contract float %379, %341
  %381 = fadd contract float %380, %373
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  store float %381, ptr addrspace(3) %281, align 4, !tbaa !27
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %382 = load float, ptr addrspace(3) %283, align 4, !tbaa !27
  %383 = fadd contract float %381, %382
  %384 = bitcast float %383 to i32
  %385 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #10
  %386 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %385) #10
  %387 = xor i32 %386, 32
  %388 = and i32 %386, -64
  %389 = add nsw i32 %388, 64
  %390 = icmp slt i32 %387, %389
  %391 = select i1 %390, i32 %387, i32 %386
  %392 = shl i32 %391, 2
  %393 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %392, i32 %384)
  %394 = bitcast i32 %393 to float
  %395 = fadd contract float %383, %394
  %396 = bitcast float %395 to i32
  %397 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #10
  %398 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %397) #10
  %399 = xor i32 %398, 16
  %400 = and i32 %398, -64
  %401 = add nsw i32 %400, 64
  %402 = icmp slt i32 %399, %401
  %403 = select i1 %402, i32 %399, i32 %398
  %404 = shl i32 %403, 2
  %405 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %404, i32 %396)
  %406 = bitcast i32 %405 to float
  %407 = fadd contract float %395, %406
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %408 = shl nuw nsw i32 %11, 6
  %409 = and i32 %408, 960
  %410 = add nuw nsw i32 %87, %15
  %411 = shl nuw nsw i32 %410, 5
  %412 = and i32 %411, 32
  %413 = add nuw nsw i32 %12, %11
  %414 = shl nuw nsw i32 %413, 3
  %415 = and i32 %414, 8
  %416 = or disjoint i32 %105, %409
  %417 = or disjoint i32 %416, %412
  %418 = or disjoint i32 %417, %415
  %419 = fdiv contract float %317, %407
  %420 = fdiv contract float %325, %407
  %421 = fdiv contract float %333, %407
  %422 = fdiv contract float %341, %407
  %423 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !29
  %424 = fptrunc float %419 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %423), !noalias !29
  %425 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !29
  %426 = fptrunc float %420 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %425), !noalias !29
  %427 = bitcast half %424 to i16
  %428 = bitcast half %426 to i16
  %429 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !34
  %430 = fptrunc float %421 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %429), !noalias !34
  %431 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !34
  %432 = fptrunc float %422 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %431), !noalias !34
  %433 = bitcast half %430 to i16
  %434 = bitcast half %432 to i16
  %435 = zext i16 %434 to i64
  %436 = shl nuw i64 %435, 48
  %437 = zext i16 %433 to i64
  %438 = shl nuw nsw i64 %437, 32
  %439 = or disjoint i64 %436, %438
  %440 = zext i16 %428 to i64
  %441 = shl nuw nsw i64 %440, 16
  %442 = or disjoint i64 %439, %441
  %443 = zext i16 %427 to i64
  %444 = or disjoint i64 %442, %443
  %445 = shl nuw nsw i32 %79, 4
  %446 = and i32 %445, 16
  %447 = or disjoint i32 %418, %446
  %448 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %447
  store i64 %444, ptr addrspace(3) %448, align 8
  %449 = fdiv contract float %349, %407
  %450 = fdiv contract float %357, %407
  %451 = fdiv contract float %365, %407
  %452 = fdiv contract float %373, %407
  %453 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !29
  %454 = fptrunc float %449 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %453), !noalias !29
  %455 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !29
  %456 = fptrunc float %450 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %455), !noalias !29
  %457 = bitcast half %454 to i16
  %458 = bitcast half %456 to i16
  %459 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !34
  %460 = fptrunc float %451 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %459), !noalias !34
  %461 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !34
  %462 = fptrunc float %452 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %461), !noalias !34
  %463 = bitcast half %460 to i16
  %464 = bitcast half %462 to i16
  %465 = zext i16 %464 to i64
  %466 = shl nuw i64 %465, 48
  %467 = zext i16 %463 to i64
  %468 = shl nuw nsw i64 %467, 32
  %469 = or disjoint i64 %466, %468
  %470 = zext i16 %458 to i64
  %471 = shl nuw nsw i64 %470, 16
  %472 = or disjoint i64 %469, %471
  %473 = zext i16 %457 to i64
  %474 = or disjoint i64 %472, %473
  %475 = xor i32 %446, 16
  %476 = or disjoint i32 %418, %475
  %477 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %476
  store i64 %474, ptr addrspace(3) %477, align 8
  %478 = or disjoint i32 %416, %415
  %479 = and i32 %69, 384
  %480 = and i32 %15, 16
  %481 = add nuw nsw i32 %480, %11
  %482 = and i32 %481, 16
  %483 = and i32 %11, 7
  %484 = shl nuw nsw i32 %410, 4
  %485 = and i32 %484, 16
  %486 = or disjoint i32 %106, %485
  %487 = or disjoint i32 %486, %104
  %488 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %487
  %489 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %85
  %490 = load i32, ptr addrspace(1) %114, align 4, !tbaa !17
  %491 = shl nsw i32 %490, 4
  %492 = or disjoint i32 %491, %89
  %493 = icmp slt i32 %490, 0
  %494 = icmp sgt i32 %492, %17
  %495 = select i1 %493, i1 true, i1 %494
  br i1 %495, label %509, label %496

496:                                              ; preds = %250
  %497 = zext nneg i32 %492 to i64
  %498 = shl nuw nsw i64 %497, 6
  %499 = add nuw nsw i64 %498, %97
  %500 = or disjoint i64 %499, %99
  %501 = getelementptr inbounds %struct.__half, ptr addrspace(4) %4, i64 %500
  %502 = load i32, ptr addrspace(4) %501, align 16, !tbaa !17
  %503 = getelementptr inbounds i8, ptr addrspace(4) %501, i64 4
  %504 = load i32, ptr addrspace(4) %503, align 4, !tbaa !17
  %505 = getelementptr inbounds i8, ptr addrspace(4) %501, i64 8
  %506 = load i32, ptr addrspace(4) %505, align 8, !tbaa !17
  %507 = getelementptr inbounds i8, ptr addrspace(4) %501, i64 12
  %508 = load i32, ptr addrspace(4) %507, align 4, !tbaa !17
  br label %509

509:                                              ; preds = %496, %250
  %510 = phi i32 [ %502, %496 ], [ 0, %250 ]
  %511 = phi i32 [ %504, %496 ], [ 0, %250 ]
  %512 = phi i32 [ %506, %496 ], [ 0, %250 ]
  %513 = phi i32 [ %508, %496 ], [ 0, %250 ]
  store i32 %510, ptr addrspace(3) %141, align 16, !tbaa !17
  store i32 %511, ptr addrspace(3) %142, align 4, !tbaa !17
  store i32 %512, ptr addrspace(3) %143, align 8, !tbaa !17
  store i32 %513, ptr addrspace(3) %144, align 4, !tbaa !17
  %514 = load i32, ptr addrspace(1) %145, align 4, !tbaa !17
  %515 = shl nsw i32 %514, 4
  %516 = or disjoint i32 %515, %89
  %517 = icmp slt i32 %514, 0
  %518 = icmp sgt i32 %516, %17
  %519 = select i1 %517, i1 true, i1 %518
  br i1 %519, label %533, label %520

520:                                              ; preds = %509
  %521 = zext nneg i32 %516 to i64
  %522 = shl nuw nsw i64 %521, 6
  %523 = add nuw nsw i64 %522, %97
  %524 = or disjoint i64 %523, %99
  %525 = getelementptr inbounds %struct.__half, ptr addrspace(4) %4, i64 %524
  %526 = load i32, ptr addrspace(4) %525, align 16, !tbaa !17
  %527 = getelementptr inbounds i8, ptr addrspace(4) %525, i64 4
  %528 = load i32, ptr addrspace(4) %527, align 4, !tbaa !17
  %529 = getelementptr inbounds i8, ptr addrspace(4) %525, i64 8
  %530 = load i32, ptr addrspace(4) %529, align 8, !tbaa !17
  %531 = getelementptr inbounds i8, ptr addrspace(4) %525, i64 12
  %532 = load i32, ptr addrspace(4) %531, align 4, !tbaa !17
  br label %533

533:                                              ; preds = %520, %509
  %534 = phi i32 [ %526, %520 ], [ 0, %509 ]
  %535 = phi i32 [ %528, %520 ], [ 0, %509 ]
  %536 = phi i32 [ %530, %520 ], [ 0, %509 ]
  %537 = phi i32 [ %532, %520 ], [ 0, %509 ]
  store i32 %534, ptr addrspace(3) %173, align 16, !tbaa !17
  store i32 %535, ptr addrspace(3) %174, align 4, !tbaa !17
  store i32 %536, ptr addrspace(3) %175, align 8, !tbaa !17
  store i32 %537, ptr addrspace(3) %176, align 4, !tbaa !17
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %538 = shl nuw nsw i32 %15, 5
  %539 = and i32 %538, 32
  %540 = or disjoint i32 %478, %446
  %541 = or disjoint i32 %540, %539
  %542 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %541
  %543 = load <4 x half>, ptr addrspace(3) %542, align 8
  %544 = or disjoint i32 %479, %482
  %545 = and i32 %11, 8
  %546 = or disjoint i32 %545, %544
  %547 = or disjoint i32 %546, %483
  %548 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %547
  %549 = load half, ptr addrspace(3) %548, align 2, !tbaa !39
  %550 = insertelement <4 x half> poison, half %549, i64 0
  %551 = or disjoint i32 %544, 32
  %552 = or disjoint i32 %545, %551
  %553 = or disjoint i32 %552, %483
  %554 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %553
  %555 = load half, ptr addrspace(3) %554, align 2, !tbaa !39
  %556 = insertelement <4 x half> %550, half %555, i64 1
  %557 = or disjoint i32 %544, 64
  %558 = xor i32 %545, 8
  %559 = or disjoint i32 %558, %557
  %560 = or disjoint i32 %559, %483
  %561 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %560
  %562 = load half, ptr addrspace(3) %561, align 2, !tbaa !39
  %563 = insertelement <4 x half> %556, half %562, i64 2
  %564 = or disjoint i32 %544, 96
  %565 = or disjoint i32 %558, %564
  %566 = or disjoint i32 %565, %483
  %567 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %566
  %568 = load half, ptr addrspace(3) %567, align 2, !tbaa !39
  %569 = insertelement <4 x half> %563, half %568, i64 3
  %570 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %569, <4 x half> %543, <4 x float> zeroinitializer)
  %571 = or disjoint i32 %478, %475
  %572 = or disjoint i32 %571, %539
  %573 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %572
  %574 = load <4 x half>, ptr addrspace(3) %573, align 8
  %575 = or disjoint i32 %544, 512
  %576 = or disjoint i32 %545, %575
  %577 = or disjoint i32 %576, %483
  %578 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %577
  %579 = load half, ptr addrspace(3) %578, align 2, !tbaa !39
  %580 = insertelement <4 x half> poison, half %579, i64 0
  %581 = or disjoint i32 %544, 544
  %582 = or disjoint i32 %545, %581
  %583 = or disjoint i32 %582, %483
  %584 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %583
  %585 = load half, ptr addrspace(3) %584, align 2, !tbaa !39
  %586 = insertelement <4 x half> %580, half %585, i64 1
  %587 = or disjoint i32 %544, 576
  %588 = or disjoint i32 %558, %587
  %589 = or disjoint i32 %588, %483
  %590 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %589
  %591 = load half, ptr addrspace(3) %590, align 2, !tbaa !39
  %592 = insertelement <4 x half> %586, half %591, i64 2
  %593 = or disjoint i32 %544, 608
  %594 = or disjoint i32 %558, %593
  %595 = or disjoint i32 %594, %483
  %596 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %595
  %597 = load half, ptr addrspace(3) %596, align 2, !tbaa !39
  %598 = insertelement <4 x half> %592, half %597, i64 3
  %599 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %598, <4 x half> %574, <4 x float> %570)
  %600 = shl nuw nsw i32 %196, 5
  %601 = and i32 %600, 32
  %602 = or disjoint i32 %540, %601
  %603 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %602
  %604 = load <4 x half>, ptr addrspace(3) %603, align 8
  %605 = or disjoint i32 %544, 1024
  %606 = or disjoint i32 %545, %605
  %607 = or disjoint i32 %606, %483
  %608 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %607
  %609 = load half, ptr addrspace(3) %608, align 2, !tbaa !39
  %610 = insertelement <4 x half> poison, half %609, i64 0
  %611 = or disjoint i32 %544, 1056
  %612 = or disjoint i32 %545, %611
  %613 = or disjoint i32 %612, %483
  %614 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %613
  %615 = load half, ptr addrspace(3) %614, align 2, !tbaa !39
  %616 = insertelement <4 x half> %610, half %615, i64 1
  %617 = or disjoint i32 %544, 1088
  %618 = or disjoint i32 %558, %617
  %619 = or disjoint i32 %618, %483
  %620 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %619
  %621 = load half, ptr addrspace(3) %620, align 2, !tbaa !39
  %622 = insertelement <4 x half> %616, half %621, i64 2
  %623 = or disjoint i32 %544, 1120
  %624 = or disjoint i32 %558, %623
  %625 = or disjoint i32 %624, %483
  %626 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %625
  %627 = load half, ptr addrspace(3) %626, align 2, !tbaa !39
  %628 = insertelement <4 x half> %622, half %627, i64 3
  %629 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %628, <4 x half> %604, <4 x float> %599)
  %630 = xor i32 %602, 16
  %631 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %630
  %632 = load <4 x half>, ptr addrspace(3) %631, align 8
  %633 = or disjoint i32 %544, 1536
  %634 = or disjoint i32 %545, %633
  %635 = or disjoint i32 %634, %483
  %636 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %635
  %637 = load half, ptr addrspace(3) %636, align 2, !tbaa !39
  %638 = insertelement <4 x half> poison, half %637, i64 0
  %639 = or disjoint i32 %544, 1568
  %640 = or disjoint i32 %545, %639
  %641 = or disjoint i32 %640, %483
  %642 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %641
  %643 = load half, ptr addrspace(3) %642, align 2, !tbaa !39
  %644 = insertelement <4 x half> %638, half %643, i64 1
  %645 = or disjoint i32 %544, 1600
  %646 = or disjoint i32 %558, %645
  %647 = or disjoint i32 %646, %483
  %648 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %647
  %649 = load half, ptr addrspace(3) %648, align 2, !tbaa !39
  %650 = insertelement <4 x half> %644, half %649, i64 2
  %651 = or disjoint i32 %544, 1632
  %652 = or disjoint i32 %558, %651
  %653 = or disjoint i32 %652, %483
  %654 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %653
  %655 = load half, ptr addrspace(3) %654, align 2, !tbaa !39
  %656 = insertelement <4 x half> %650, half %655, i64 3
  %657 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %656, <4 x half> %632, <4 x float> %629)
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %658 = extractelement <4 x float> %657, i64 0
  %659 = extractelement <4 x float> %657, i64 1
  %660 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !40
  %661 = fptrunc float %658 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %660), !noalias !40
  %662 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !40
  %663 = fptrunc float %659 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %662), !noalias !40
  %664 = bitcast half %661 to i16
  %665 = bitcast half %663 to i16
  %666 = extractelement <4 x float> %657, i64 2
  %667 = extractelement <4 x float> %657, i64 3
  %668 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !45
  %669 = fptrunc float %666 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %668), !noalias !45
  %670 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !45
  %671 = fptrunc float %667 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %670), !noalias !45
  %672 = bitcast half %669 to i16
  %673 = bitcast half %671 to i16
  %674 = zext i16 %673 to i64
  %675 = shl nuw i64 %674, 48
  %676 = zext i16 %672 to i64
  %677 = shl nuw nsw i64 %676, 32
  %678 = or disjoint i64 %675, %677
  %679 = zext i16 %665 to i64
  %680 = shl nuw nsw i64 %679, 16
  %681 = or disjoint i64 %678, %680
  %682 = zext i16 %664 to i64
  %683 = or disjoint i64 %681, %682
  store i64 %683, ptr addrspace(3) %488, align 8
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %684 = getelementptr inbounds %struct.__half, ptr addrspace(1) %2, i64 %111
  %685 = load i64, ptr addrspace(3) %489, align 8
  store i64 %685, ptr addrspace(1) %684, align 8
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %686 = load i32, ptr addrspace(1) %114, align 4, !tbaa !17
  %687 = shl nsw i32 %686, 4
  %688 = or disjoint i32 %687, %89
  %689 = icmp slt i32 %686, 0
  %690 = icmp sgt i32 %688, %17
  %691 = select i1 %689, i1 true, i1 %690
  br i1 %691, label %706, label %692

692:                                              ; preds = %533
  %693 = zext nneg i32 %688 to i64
  %694 = shl nuw nsw i64 %693, 6
  %695 = add nuw nsw i64 %694, %97
  %696 = or disjoint i64 %695, %99
  %697 = or disjoint i64 %696, 32
  %698 = getelementptr inbounds %struct.__half, ptr addrspace(4) %4, i64 %697
  %699 = load i32, ptr addrspace(4) %698, align 16, !tbaa !17
  %700 = getelementptr inbounds i8, ptr addrspace(4) %698, i64 4
  %701 = load i32, ptr addrspace(4) %700, align 4, !tbaa !17
  %702 = getelementptr inbounds i8, ptr addrspace(4) %698, i64 8
  %703 = load i32, ptr addrspace(4) %702, align 8, !tbaa !17
  %704 = getelementptr inbounds i8, ptr addrspace(4) %698, i64 12
  %705 = load i32, ptr addrspace(4) %704, align 4, !tbaa !17
  br label %706

706:                                              ; preds = %692, %533
  %707 = phi i32 [ %699, %692 ], [ 0, %533 ]
  %708 = phi i32 [ %701, %692 ], [ 0, %533 ]
  %709 = phi i32 [ %703, %692 ], [ 0, %533 ]
  %710 = phi i32 [ %705, %692 ], [ 0, %533 ]
  store i32 %707, ptr addrspace(3) %141, align 16, !tbaa !17
  store i32 %708, ptr addrspace(3) %142, align 4, !tbaa !17
  store i32 %709, ptr addrspace(3) %143, align 8, !tbaa !17
  store i32 %710, ptr addrspace(3) %144, align 4, !tbaa !17
  %711 = load i32, ptr addrspace(1) %145, align 4, !tbaa !17
  %712 = shl nsw i32 %711, 4
  %713 = or disjoint i32 %712, %89
  %714 = icmp slt i32 %711, 0
  %715 = icmp sgt i32 %713, %17
  %716 = select i1 %714, i1 true, i1 %715
  br i1 %716, label %731, label %717

717:                                              ; preds = %706
  %718 = zext nneg i32 %713 to i64
  %719 = shl nuw nsw i64 %718, 6
  %720 = add nuw nsw i64 %719, %97
  %721 = or disjoint i64 %720, %99
  %722 = or disjoint i64 %721, 32
  %723 = getelementptr inbounds %struct.__half, ptr addrspace(4) %4, i64 %722
  %724 = load i32, ptr addrspace(4) %723, align 16, !tbaa !17
  %725 = getelementptr inbounds i8, ptr addrspace(4) %723, i64 4
  %726 = load i32, ptr addrspace(4) %725, align 4, !tbaa !17
  %727 = getelementptr inbounds i8, ptr addrspace(4) %723, i64 8
  %728 = load i32, ptr addrspace(4) %727, align 8, !tbaa !17
  %729 = getelementptr inbounds i8, ptr addrspace(4) %723, i64 12
  %730 = load i32, ptr addrspace(4) %729, align 4, !tbaa !17
  br label %731

731:                                              ; preds = %717, %706
  %732 = phi i32 [ %724, %717 ], [ 0, %706 ]
  %733 = phi i32 [ %726, %717 ], [ 0, %706 ]
  %734 = phi i32 [ %728, %717 ], [ 0, %706 ]
  %735 = phi i32 [ %730, %717 ], [ 0, %706 ]
  store i32 %732, ptr addrspace(3) %173, align 16, !tbaa !17
  store i32 %733, ptr addrspace(3) %174, align 4, !tbaa !17
  store i32 %734, ptr addrspace(3) %175, align 8, !tbaa !17
  store i32 %735, ptr addrspace(3) %176, align 4, !tbaa !17
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %736 = load <4 x half>, ptr addrspace(3) %542, align 8
  %737 = load half, ptr addrspace(3) %548, align 2, !tbaa !39
  %738 = insertelement <4 x half> poison, half %737, i64 0
  %739 = load half, ptr addrspace(3) %554, align 2, !tbaa !39
  %740 = insertelement <4 x half> %738, half %739, i64 1
  %741 = load half, ptr addrspace(3) %561, align 2, !tbaa !39
  %742 = insertelement <4 x half> %740, half %741, i64 2
  %743 = load half, ptr addrspace(3) %567, align 2, !tbaa !39
  %744 = insertelement <4 x half> %742, half %743, i64 3
  %745 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %744, <4 x half> %736, <4 x float> zeroinitializer)
  %746 = load <4 x half>, ptr addrspace(3) %573, align 8
  %747 = load half, ptr addrspace(3) %578, align 2, !tbaa !39
  %748 = insertelement <4 x half> poison, half %747, i64 0
  %749 = load half, ptr addrspace(3) %584, align 2, !tbaa !39
  %750 = insertelement <4 x half> %748, half %749, i64 1
  %751 = load half, ptr addrspace(3) %590, align 2, !tbaa !39
  %752 = insertelement <4 x half> %750, half %751, i64 2
  %753 = load half, ptr addrspace(3) %596, align 2, !tbaa !39
  %754 = insertelement <4 x half> %752, half %753, i64 3
  %755 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %754, <4 x half> %746, <4 x float> %745)
  %756 = load <4 x half>, ptr addrspace(3) %603, align 8
  %757 = load half, ptr addrspace(3) %608, align 2, !tbaa !39
  %758 = insertelement <4 x half> poison, half %757, i64 0
  %759 = load half, ptr addrspace(3) %614, align 2, !tbaa !39
  %760 = insertelement <4 x half> %758, half %759, i64 1
  %761 = load half, ptr addrspace(3) %620, align 2, !tbaa !39
  %762 = insertelement <4 x half> %760, half %761, i64 2
  %763 = load half, ptr addrspace(3) %626, align 2, !tbaa !39
  %764 = insertelement <4 x half> %762, half %763, i64 3
  %765 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %764, <4 x half> %756, <4 x float> %755)
  %766 = load <4 x half>, ptr addrspace(3) %631, align 8
  %767 = load half, ptr addrspace(3) %636, align 2, !tbaa !39
  %768 = insertelement <4 x half> poison, half %767, i64 0
  %769 = load half, ptr addrspace(3) %642, align 2, !tbaa !39
  %770 = insertelement <4 x half> %768, half %769, i64 1
  %771 = load half, ptr addrspace(3) %648, align 2, !tbaa !39
  %772 = insertelement <4 x half> %770, half %771, i64 2
  %773 = load half, ptr addrspace(3) %654, align 2, !tbaa !39
  %774 = insertelement <4 x half> %772, half %773, i64 3
  %775 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %774, <4 x half> %766, <4 x float> %765)
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %776 = extractelement <4 x float> %775, i64 0
  %777 = extractelement <4 x float> %775, i64 1
  %778 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !40
  %779 = fptrunc float %776 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %778), !noalias !40
  %780 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !40
  %781 = fptrunc float %777 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %780), !noalias !40
  %782 = bitcast half %779 to i16
  %783 = bitcast half %781 to i16
  %784 = extractelement <4 x float> %775, i64 2
  %785 = extractelement <4 x float> %775, i64 3
  %786 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !45
  %787 = fptrunc float %784 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %786), !noalias !45
  %788 = tail call i32 @llvm.mxc.gethwreg(i32 2177)
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 0), !noalias !45
  %789 = fptrunc float %785 to half
  tail call void @llvm.mxc.sethwreg(i32 2177, i32 %788), !noalias !45
  %790 = bitcast half %787 to i16
  %791 = bitcast half %789 to i16
  %792 = zext i16 %791 to i64
  %793 = shl nuw i64 %792, 48
  %794 = zext i16 %790 to i64
  %795 = shl nuw nsw i64 %794, 32
  %796 = or disjoint i64 %793, %795
  %797 = zext i16 %783 to i64
  %798 = shl nuw nsw i64 %797, 16
  %799 = or disjoint i64 %796, %798
  %800 = zext i16 %782 to i64
  %801 = or disjoint i64 %799, %800
  store i64 %801, ptr addrspace(3) %488, align 8
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %802 = getelementptr inbounds %struct.__half, ptr addrspace(1) %2, i64 %214
  %803 = load i64, ptr addrspace(3) %489, align 8
  store i64 %803, ptr addrspace(1) %802, align 8
  ret void
}

; Function Attrs: nounwind speculatable willreturn memory(none)
declare i32 @llvm.mxc.thread.id.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #6

; Function Attrs: convergent nounwind willreturn
declare void @llvm.mxc.barrier() #7

; Function Attrs: convergent nounwind willreturn memory(none)
declare <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>) #8

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #6

; Function Attrs: convergent nounwind willreturn memory(none)
declare i32 @llvm.mxc.mbcnt.lo(i32, i32) #8

; Function Attrs: convergent nounwind willreturn memory(none)
declare i32 @llvm.mxc.mbcnt.hi(i32, i32) #8

; Function Attrs: convergent nounwind willreturn memory(none)
declare i32 @llvm.mxc.bsm.bpermute(i32, i32) #8

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp2.f32(float) #6

; Function Attrs: nounwind speculatable willreturn memory(inaccessiblemem: read)
declare i32 @llvm.mxc.gethwreg(i32 immarg) #9

; Function Attrs: nounwind willreturn
declare void @llvm.mxc.sethwreg(i32 immarg, i32) #4

attributes #0 = { mustprogress noreturn nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" "disable-promote-alloca-to-bsm"="true" "disable-promote-alloca-to-vector"="false" "enable-ldg-bsm-opt"="false" "fixed-function-abi"="true" "metaxgpu-bsm-direct-address"="true" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="512" "metaxgpu-new-streg-abi"="false" "metaxgpu-pk-fma"="false" "metaxgpu-resource-usage"="false" "metaxgpu-sched-select"="default" "metaxgpu-use-dim-intrinsic"="false" "no-trapping-math"="true" "prec-div"="false" "prec-sqrt"="false" "scalarize-global-loads"="true" "shfl-combine"="true" "stack-protector-buffer-size"="8" "target-cpu"="xcore1000" "target-features"="+xcore1000" }
attributes #1 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #2 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" "disable-promote-alloca-to-bsm"="true" "disable-promote-alloca-to-vector"="false" "enable-ldg-bsm-opt"="false" "fixed-function-abi"="true" "metaxgpu-bsm-direct-address"="true" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="512" "metaxgpu-new-streg-abi"="false" "metaxgpu-pk-fma"="false" "metaxgpu-resource-usage"="false" "metaxgpu-sched-select"="default" "metaxgpu-use-dim-intrinsic"="false" "no-trapping-math"="true" "prec-div"="false" "prec-sqrt"="false" "scalarize-global-loads"="true" "shfl-combine"="true" "stack-protector-buffer-size"="8" "target-cpu"="xcore1000" "target-features"="+xcore1000" }
attributes #3 = { nounwind speculatable willreturn memory(none) }
attributes #4 = { nounwind willreturn }
attributes #5 = { convergent mustprogress norecurse nounwind willreturn "denormal-fp-math-f32"="preserve-sign,preserve-sign" "disable-promote-alloca-to-bsm"="true" "disable-promote-alloca-to-vector"="false" "enable-ldg-bsm-opt"="false" "fixed-function-abi"="true" "metaxgpu-bsm-direct-address"="true" "metaxgpu-implicitarg-num-bytes"="80" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="128" "metaxgpu-min-blocks"="1" "metaxgpu-new-streg-abi"="false" "metaxgpu-pk-fma"="false" "metaxgpu-resource-usage"="false" "metaxgpu-sched-select"="default" "metaxgpu-use-dim-intrinsic"="false" "no-trapping-math"="true" "prec-div"="false" "prec-sqrt"="false" "scalarize-global-loads"="true" "shfl-combine"="true" "stack-protector-buffer-size"="8" "target-cpu"="xcore1000" "target-features"="+xcore1000" "uniform-work-group-size"="true" }
attributes #6 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { convergent nounwind willreturn }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { nounwind speculatable willreturn memory(inaccessiblemem: read) }
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
!27 = !{!28, !28, i64 0}
!28 = !{!"float", !5, i64 0}
!29 = !{!30, !32}
!30 = distinct !{!30, !31, !"_ZL17__floats2half2_rnff: %agg.result"}
!31 = distinct !{!31, !"_ZL17__floats2half2_rnff"}
!32 = distinct !{!32, !33, !"_ZL17__float22half2_rn6float2: %agg.result"}
!33 = distinct !{!33, !"_ZL17__float22half2_rn6float2"}
!34 = !{!35, !37}
!35 = distinct !{!35, !36, !"_ZL17__floats2half2_rnff: %agg.result"}
!36 = distinct !{!36, !"_ZL17__floats2half2_rnff"}
!37 = distinct !{!37, !38, !"_ZL17__float22half2_rn6float2: %agg.result"}
!38 = distinct !{!38, !"_ZL17__float22half2_rn6float2"}
!39 = !{!12, !12, i64 0}
!40 = !{!41, !43}
!41 = distinct !{!41, !42, !"_ZL17__floats2half2_rnff: %agg.result"}
!42 = distinct !{!42, !"_ZL17__floats2half2_rnff"}
!43 = distinct !{!43, !44, !"_ZL17__float22half2_rn6float2: %agg.result"}
!44 = distinct !{!44, !"_ZL17__float22half2_rn6float2"}
!45 = !{!46, !48}
!46 = distinct !{!46, !47, !"_ZL17__floats2half2_rnff: %agg.result"}
!47 = distinct !{!47, !"_ZL17__floats2half2_rnff"}
!48 = distinct !{!48, !49, !"_ZL17__float22half2_rn6float2: %agg.result"}
!49 = distinct !{!49, !"_ZL17__float22half2_rn6float2"}
