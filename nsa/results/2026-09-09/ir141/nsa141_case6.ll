; ModuleID = '/data/nsa_codex_20260909_r1/results/ir141/nsa141_case6.bc'
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
  br i1 %14, label %636, label %15

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
  %34 = icmp ult i32 %12, 32
  br i1 %34, label %53, label %35

35:                                               ; preds = %15
  %36 = lshr i32 %16, 3
  %37 = add i32 %13, %36
  %38 = ashr i32 %37, 4
  %39 = zext nneg i32 %6 to i64
  %40 = shl nuw nsw i64 %39, 17
  %41 = shl nuw nsw i32 %16, 4
  %42 = and i32 %41, 16256
  %43 = zext nneg i32 %42 to i64
  %44 = sext i32 %13 to i64
  %45 = or disjoint i64 %40, %43
  %46 = shl nuw nsw i32 %16, 3
  %47 = and i32 %46, 56
  %48 = zext nneg i32 %47 to i64
  %49 = or disjoint i64 %45, %48
  %50 = icmp slt i32 %38, 64
  %51 = icmp sgt i32 %37, -1
  %52 = and i1 %50, %51
  br i1 %52, label %87, label %98

53:                                               ; preds = %15
  %54 = shl nsw i32 %6, 17
  %55 = shl nuw nsw i32 %16, 4
  %56 = and i32 %55, 16256
  %57 = shl nuw nsw i32 %12, 12
  %58 = or disjoint i32 %56, %54
  %59 = shl nuw nsw i32 %16, 3
  %60 = and i32 %59, 56
  %61 = or disjoint i32 %58, %60
  %62 = add nuw nsw i32 %61, %57
  %63 = zext nneg i32 %62 to i64
  %64 = getelementptr inbounds %struct.__half, ptr addrspace(4) %4, i64 %63
  %65 = load i32, ptr addrspace(4) %64, align 16, !tbaa !17
  %66 = getelementptr inbounds i8, ptr addrspace(4) %64, i64 4
  %67 = load i32, ptr addrspace(4) %66, align 4, !tbaa !17
  %68 = getelementptr inbounds i8, ptr addrspace(4) %64, i64 8
  %69 = load i32, ptr addrspace(4) %68, align 8, !tbaa !17
  %70 = getelementptr inbounds i8, ptr addrspace(4) %64, i64 12
  %71 = load i32, ptr addrspace(4) %70, align 4, !tbaa !17
  %72 = getelementptr inbounds i8, ptr addrspace(4) %64, i64 4096
  %73 = load i32, ptr addrspace(4) %72, align 16, !tbaa !17
  %74 = getelementptr inbounds i8, ptr addrspace(4) %64, i64 4100
  %75 = load i32, ptr addrspace(4) %74, align 4, !tbaa !17
  %76 = getelementptr inbounds i8, ptr addrspace(4) %64, i64 4104
  %77 = load i32, ptr addrspace(4) %76, align 8, !tbaa !17
  %78 = getelementptr inbounds i8, ptr addrspace(4) %64, i64 4108
  %79 = load i32, ptr addrspace(4) %78, align 4, !tbaa !17
  %80 = zext nneg i32 %6 to i64
  %81 = shl nuw nsw i64 %80, 17
  %82 = zext nneg i32 %56 to i64
  %83 = zext nneg i32 %13 to i64
  %84 = or disjoint i64 %81, %82
  %85 = zext nneg i32 %60 to i64
  %86 = or disjoint i64 %84, %85
  br label %119

87:                                               ; preds = %35
  %88 = getelementptr %struct.__half, ptr addrspace(4) %4, i64 %49
  %89 = shl nsw i64 %44, 8
  %90 = getelementptr i8, ptr addrspace(4) %88, i64 %89
  %91 = load i32, ptr addrspace(4) %90, align 16, !tbaa !17
  %92 = getelementptr inbounds i8, ptr addrspace(4) %90, i64 4
  %93 = load i32, ptr addrspace(4) %92, align 4, !tbaa !17
  %94 = getelementptr inbounds i8, ptr addrspace(4) %90, i64 8
  %95 = load i32, ptr addrspace(4) %94, align 8, !tbaa !17
  %96 = getelementptr inbounds i8, ptr addrspace(4) %90, i64 12
  %97 = load i32, ptr addrspace(4) %96, align 4, !tbaa !17
  br label %98

98:                                               ; preds = %87, %35
  %99 = phi i32 [ %91, %87 ], [ 0, %35 ]
  %100 = phi i32 [ %93, %87 ], [ 0, %35 ]
  %101 = phi i32 [ %95, %87 ], [ 0, %35 ]
  %102 = phi i32 [ %97, %87 ], [ 0, %35 ]
  %103 = icmp slt i32 %38, 63
  %104 = add i32 %37, 16
  %105 = icmp sgt i32 %104, -1
  %106 = and i1 %103, %105
  br i1 %106, label %107, label %119

107:                                              ; preds = %98
  %108 = getelementptr %struct.__half, ptr addrspace(4) %4, i64 %49
  %109 = shl nsw i64 %44, 8
  %110 = getelementptr i8, ptr addrspace(4) %108, i64 %109
  %111 = getelementptr i8, ptr addrspace(4) %110, i64 4096
  %112 = load i32, ptr addrspace(4) %111, align 16, !tbaa !17
  %113 = getelementptr i8, ptr addrspace(4) %110, i64 4100
  %114 = load i32, ptr addrspace(4) %113, align 4, !tbaa !17
  %115 = getelementptr i8, ptr addrspace(4) %110, i64 4104
  %116 = load i32, ptr addrspace(4) %115, align 8, !tbaa !17
  %117 = getelementptr i8, ptr addrspace(4) %110, i64 4108
  %118 = load i32, ptr addrspace(4) %117, align 4, !tbaa !17
  br label %119

119:                                              ; preds = %107, %98, %53
  %120 = phi i64 [ %86, %53 ], [ %49, %107 ], [ %49, %98 ]
  %121 = phi i64 [ %83, %53 ], [ %44, %107 ], [ %44, %98 ]
  %122 = phi i32 [ %60, %53 ], [ %47, %107 ], [ %47, %98 ]
  %123 = phi i32 [ %59, %53 ], [ %46, %107 ], [ %46, %98 ]
  %124 = phi i32 [ %56, %53 ], [ %42, %107 ], [ %42, %98 ]
  %125 = phi i32 [ %55, %53 ], [ %41, %107 ], [ %41, %98 ]
  %126 = phi i32 [ %73, %53 ], [ %112, %107 ], [ 0, %98 ]
  %127 = phi i32 [ %71, %53 ], [ %102, %107 ], [ %102, %98 ]
  %128 = phi i32 [ %69, %53 ], [ %101, %107 ], [ %101, %98 ]
  %129 = phi i32 [ %67, %53 ], [ %100, %107 ], [ %100, %98 ]
  %130 = phi i32 [ %65, %53 ], [ %99, %107 ], [ %99, %98 ]
  %131 = phi i32 [ %75, %53 ], [ %114, %107 ], [ 0, %98 ]
  %132 = phi i32 [ %77, %53 ], [ %116, %107 ], [ 0, %98 ]
  %133 = phi i32 [ %79, %53 ], [ %118, %107 ], [ 0, %98 ]
  %134 = shl nsw i32 %6, 21
  %135 = shl nsw i32 %8, 11
  %136 = add nuw nsw i32 %134, %135
  %137 = add nuw nsw i32 %136, %124
  %138 = or disjoint i32 %137, %122
  %139 = and i32 %123, 8128
  %140 = and i32 %123, 32
  %141 = add nuw nsw i32 %140, %16
  %142 = and i32 %141, 32
  %143 = and i32 %123, 16
  %144 = add nuw nsw i32 %143, %16
  %145 = and i32 %144, 16
  %146 = mul nuw nsw i32 %16, 9
  %147 = and i32 %146, 8
  %148 = or disjoint i32 %142, %147
  %149 = or disjoint i32 %148, %139
  %150 = or disjoint i32 %149, %145
  %151 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %150
  %152 = shl nuw nsw i32 %16, 6
  %153 = and i32 %152, 960
  %154 = lshr i32 %16, 1
  %155 = lshr i32 %16, 5
  %156 = add nuw nsw i32 %155, %16
  %157 = shl nuw nsw i32 %156, 3
  %158 = and i32 %157, 8
  %159 = and i32 %17, 4
  %160 = or disjoint i32 %159, %153
  %161 = or disjoint i32 %160, %158
  %162 = and i32 %125, 15360
  %163 = or disjoint i32 %153, %162
  %164 = or disjoint i32 %163, %159
  %165 = or disjoint i32 %164, %158
  %166 = zext nneg i32 %138 to i64
  %167 = getelementptr inbounds %struct.__half, ptr addrspace(4) %3, i64 %166
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %151, ptr addrspace(4) noundef align 16 dereferenceable(16) %167, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %168 = getelementptr %struct.__half, ptr addrspace(4) %1, i64 %120
  %169 = shl nsw i64 %121, 8
  %170 = getelementptr i8, ptr addrspace(4) %168, i64 %169
  %171 = or disjoint i32 %139, %142
  %172 = or disjoint i32 %171, %145
  %173 = or disjoint i32 %172, %147
  %174 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %173
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %174, ptr addrspace(4) noundef align 16 dereferenceable(16) %170, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  %175 = getelementptr i8, ptr addrspace(4) %170, i64 4096
  %176 = add nuw nsw i32 %139, 1024
  %177 = or disjoint i32 %176, %142
  %178 = or disjoint i32 %177, %145
  %179 = or disjoint i32 %178, %147
  %180 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %179
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %180, ptr addrspace(4) noundef align 16 dereferenceable(16) %175, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %181 = shl nuw nsw i32 %17, 5
  %182 = and i32 %181, 32
  %183 = shl nuw nsw i32 %154, 4
  %184 = and i32 %183, 16
  %185 = or disjoint i32 %161, %184
  %186 = or disjoint i32 %185, %182
  %187 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %186
  %188 = load <4 x half>, ptr addrspace(3) %187, align 8
  %189 = or disjoint i32 %165, %184
  %190 = or disjoint i32 %189, %182
  %191 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %190
  %192 = load <4 x half>, ptr addrspace(3) %191, align 8
  %193 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %192, <4 x half> %188, <4 x float> %33)
  %194 = xor i32 %184, 16
  %195 = or disjoint i32 %161, %194
  %196 = or disjoint i32 %195, %182
  %197 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %196
  %198 = load <4 x half>, ptr addrspace(3) %197, align 8
  %199 = or disjoint i32 %165, %194
  %200 = or disjoint i32 %199, %182
  %201 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %200
  %202 = load <4 x half>, ptr addrspace(3) %201, align 8
  %203 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %202, <4 x half> %198, <4 x float> %193)
  %204 = add nuw nsw i32 %17, 1
  %205 = shl nuw nsw i32 %204, 5
  %206 = and i32 %205, 32
  %207 = or disjoint i32 %185, %206
  %208 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %207
  %209 = load <4 x half>, ptr addrspace(3) %208, align 8
  %210 = or disjoint i32 %189, %206
  %211 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %210
  %212 = load <4 x half>, ptr addrspace(3) %211, align 8
  %213 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %212, <4 x half> %209, <4 x float> %203)
  %214 = or disjoint i32 %195, %206
  %215 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 4096), i32 %214
  %216 = load <4 x half>, ptr addrspace(3) %215, align 8
  %217 = or disjoint i32 %199, %206
  %218 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %217
  %219 = load <4 x half>, ptr addrspace(3) %218, align 8
  %220 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %219, <4 x half> %216, <4 x float> %213)
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %221 = or disjoint i64 %166, 64
  %222 = getelementptr inbounds %struct.__half, ptr addrspace(4) %3, i64 %221
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %151, ptr addrspace(4) noundef align 16 dereferenceable(16) %222, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !28
  %223 = getelementptr i8, ptr addrspace(4) %170, i64 128
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %174, ptr addrspace(4) noundef align 16 dereferenceable(16) %223, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  %224 = getelementptr i8, ptr addrspace(4) %170, i64 4224
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %180, ptr addrspace(4) noundef align 16 dereferenceable(16) %224, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !29
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %225 = load <4 x half>, ptr addrspace(3) %187, align 8
  %226 = load <4 x half>, ptr addrspace(3) %191, align 8
  %227 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %226, <4 x half> %225, <4 x float> %220)
  %228 = load <4 x half>, ptr addrspace(3) %197, align 8
  %229 = load <4 x half>, ptr addrspace(3) %201, align 8
  %230 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %229, <4 x half> %228, <4 x float> %227)
  %231 = load <4 x half>, ptr addrspace(3) %208, align 8
  %232 = load <4 x half>, ptr addrspace(3) %211, align 8
  %233 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %232, <4 x half> %231, <4 x float> %230)
  %234 = load <4 x half>, ptr addrspace(3) %215, align 8
  %235 = load <4 x half>, ptr addrspace(3) %218, align 8
  %236 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %235, <4 x half> %234, <4 x float> %233)
  %237 = extractelement <4 x float> %236, i64 0
  %238 = tail call contract noundef float @llvm.maxnum.f32(float %237, float 0xFFF0000000000000)
  %239 = extractelement <4 x float> %236, i64 1
  %240 = tail call contract noundef float @llvm.maxnum.f32(float %238, float %239)
  %241 = extractelement <4 x float> %236, i64 2
  %242 = tail call contract noundef float @llvm.maxnum.f32(float %240, float %241)
  %243 = extractelement <4 x float> %236, i64 3
  %244 = tail call contract noundef float @llvm.maxnum.f32(float %242, float %243)
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %245 = getelementptr inbounds float, ptr addrspace(3) @buf_dyn_shmem, i32 %16
  store float %244, ptr addrspace(3) %245, align 4, !tbaa !30
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %246 = xor i32 %16, 64
  %247 = getelementptr inbounds float, ptr addrspace(3) @buf_dyn_shmem, i32 %246
  %248 = load float, ptr addrspace(3) %247, align 4, !tbaa !30
  %249 = tail call contract noundef float @llvm.maxnum.f32(float %244, float %248)
  %250 = bitcast float %249 to i32
  %251 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #10
  %252 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %251) #10
  %253 = xor i32 %252, 32
  %254 = and i32 %252, -64
  %255 = add nsw i32 %254, 64
  %256 = icmp slt i32 %253, %255
  %257 = select i1 %256, i32 %253, i32 %252
  %258 = shl i32 %257, 2
  %259 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %258, i32 %250)
  %260 = bitcast i32 %259 to float
  %261 = tail call contract noundef float @llvm.maxnum.f32(float %249, float %260)
  %262 = bitcast float %261 to i32
  %263 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #10
  %264 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %263) #10
  %265 = xor i32 %264, 16
  %266 = and i32 %264, -64
  %267 = add nsw i32 %266, 64
  %268 = icmp slt i32 %265, %267
  %269 = select i1 %268, i32 %265, i32 %264
  %270 = shl i32 %269, 2
  %271 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %270, i32 %262)
  %272 = bitcast i32 %271 to float
  %273 = tail call contract noundef float @llvm.maxnum.f32(float %261, float %272)
  %274 = fsub contract float %237, %273
  %275 = fmul contract float %274, 0x3FC0527DC0000000
  %276 = fcmp contract olt float %275, -1.260000e+02
  %277 = select contract i1 %276, float 6.400000e+01, float 0.000000e+00
  %278 = fadd contract float %275, %277
  %279 = tail call contract float @llvm.exp2.f32(float %278)
  %280 = select contract i1 %276, float 0x3BF0000000000000, float 1.000000e+00
  %281 = fmul contract float %280, %279
  %282 = fsub contract float %239, %273
  %283 = fmul contract float %282, 0x3FC0527DC0000000
  %284 = fcmp contract olt float %283, -1.260000e+02
  %285 = select contract i1 %284, float 6.400000e+01, float 0.000000e+00
  %286 = fadd contract float %283, %285
  %287 = tail call contract float @llvm.exp2.f32(float %286)
  %288 = select contract i1 %284, float 0x3BF0000000000000, float 1.000000e+00
  %289 = fmul contract float %288, %287
  %290 = fsub contract float %241, %273
  %291 = fmul contract float %290, 0x3FC0527DC0000000
  %292 = fcmp contract olt float %291, -1.260000e+02
  %293 = select contract i1 %292, float 6.400000e+01, float 0.000000e+00
  %294 = fadd contract float %291, %293
  %295 = tail call contract float @llvm.exp2.f32(float %294)
  %296 = select contract i1 %292, float 0x3BF0000000000000, float 1.000000e+00
  %297 = fmul contract float %296, %295
  %298 = fsub contract float %243, %273
  %299 = fmul contract float %298, 0x3FC0527DC0000000
  %300 = fcmp contract olt float %299, -1.260000e+02
  %301 = select contract i1 %300, float 6.400000e+01, float 0.000000e+00
  %302 = fadd contract float %299, %301
  %303 = tail call contract float @llvm.exp2.f32(float %302)
  %304 = select contract i1 %300, float 0x3BF0000000000000, float 1.000000e+00
  %305 = fmul contract float %304, %303
  %306 = fadd contract float %281, 0.000000e+00
  %307 = fadd contract float %306, %289
  %308 = fadd contract float %307, %297
  %309 = fadd contract float %308, %305
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  store float %309, ptr addrspace(3) %245, align 4, !tbaa !30
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %310 = load float, ptr addrspace(3) %247, align 4, !tbaa !30
  %311 = fadd contract float %309, %310
  %312 = bitcast float %311 to i32
  %313 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #10
  %314 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %313) #10
  %315 = xor i32 %314, 32
  %316 = and i32 %314, -64
  %317 = add nsw i32 %316, 64
  %318 = icmp slt i32 %315, %317
  %319 = select i1 %318, i32 %315, i32 %314
  %320 = shl i32 %319, 2
  %321 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %320, i32 %312)
  %322 = bitcast i32 %321 to float
  %323 = fadd contract float %311, %322
  %324 = bitcast float %323 to i32
  %325 = tail call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) #10
  %326 = tail call noundef i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %325) #10
  %327 = xor i32 %326, 16
  %328 = and i32 %326, -64
  %329 = add nsw i32 %328, 64
  %330 = icmp slt i32 %327, %329
  %331 = select i1 %330, i32 %327, i32 %326
  %332 = shl i32 %331, 2
  %333 = tail call noundef i32 @llvm.mxc.bsm.bpermute(i32 %332, i32 %324)
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %334 = bitcast i32 %333 to float
  %335 = fadd contract float %323, %334
  %336 = fdiv contract float %305, %335
  %337 = fdiv contract float %297, %335
  %338 = fdiv contract float %289, %335
  %339 = fdiv contract float %281, %335
  %340 = fptrunc float %339 to half
  %341 = fptrunc float %338 to half
  %342 = fptrunc float %337 to half
  %343 = fptrunc float %336 to half
  %344 = shl nuw nsw i32 %16, 5
  %345 = and i32 %344, 480
  %346 = lshr i32 %16, 6
  %347 = add nuw nsw i32 %346, %17
  %348 = shl nuw nsw i32 %347, 4
  %349 = and i32 %348, 16
  %350 = add nuw nsw i32 %155, %154
  %351 = shl nuw nsw i32 %350, 3
  %352 = and i32 %351, 8
  %353 = or disjoint i32 %345, %349
  %354 = or disjoint i32 %353, %159
  %355 = or disjoint i32 %354, %352
  %356 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %355
  store half %340, ptr addrspace(3) %356, align 8
  %357 = getelementptr inbounds i8, ptr addrspace(3) %356, i32 2
  store half %341, ptr addrspace(3) %357, align 2
  %358 = getelementptr inbounds i8, ptr addrspace(3) %356, i32 4
  store half %342, ptr addrspace(3) %358, align 4
  %359 = getelementptr inbounds i8, ptr addrspace(3) %356, i32 6
  store half %343, ptr addrspace(3) %359, align 2
  %360 = and i32 %125, 768
  %361 = lshr i32 %16, 4
  %362 = add nuw nsw i32 %346, %361
  %363 = shl nuw nsw i32 %362, 5
  %364 = and i32 %363, 32
  %365 = and i32 %16, 7
  %366 = lshr i32 %16, 3
  %367 = add i32 %13, %366
  %368 = ashr i32 %367, 4
  %369 = or disjoint i64 %120, 64
  %370 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %173
  store i32 %130, ptr addrspace(3) %370, align 16, !tbaa !17
  %371 = getelementptr inbounds i8, ptr addrspace(3) %370, i32 4
  store i32 %129, ptr addrspace(3) %371, align 4, !tbaa !17
  %372 = getelementptr inbounds i8, ptr addrspace(3) %370, i32 8
  store i32 %128, ptr addrspace(3) %372, align 8, !tbaa !17
  %373 = getelementptr inbounds i8, ptr addrspace(3) %370, i32 12
  store i32 %127, ptr addrspace(3) %373, align 4, !tbaa !17
  %374 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %179
  store i32 %126, ptr addrspace(3) %374, align 16, !tbaa !17
  %375 = getelementptr inbounds i8, ptr addrspace(3) %374, i32 4
  store i32 %131, ptr addrspace(3) %375, align 4, !tbaa !17
  %376 = getelementptr inbounds i8, ptr addrspace(3) %374, i32 8
  store i32 %132, ptr addrspace(3) %376, align 8, !tbaa !17
  %377 = getelementptr inbounds i8, ptr addrspace(3) %374, i32 12
  store i32 %133, ptr addrspace(3) %377, align 4, !tbaa !17
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %378 = shl nuw nsw i32 %17, 4
  %379 = and i32 %378, 16
  %380 = or disjoint i32 %345, %379
  %381 = or disjoint i32 %380, %159
  %382 = or disjoint i32 %381, %352
  %383 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %382
  %384 = load <4 x half>, ptr addrspace(3) %383, align 8
  %385 = or disjoint i32 %360, %364
  %386 = and i32 %16, 8
  %387 = or disjoint i32 %385, %386
  %388 = or disjoint i32 %387, %365
  %389 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %388
  %390 = load half, ptr addrspace(3) %389, align 2, !tbaa !32
  %391 = insertelement <4 x half> poison, half %390, i64 0
  %392 = or disjoint i32 %386, %385
  %393 = or disjoint i32 %392, %365
  %394 = xor i32 %393, 8
  %395 = or disjoint i32 %394, 64
  %396 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %395
  %397 = load half, ptr addrspace(3) %396, align 2, !tbaa !32
  %398 = insertelement <4 x half> %391, half %397, i64 1
  %399 = or disjoint i32 %388, 144
  %400 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %399
  %401 = load half, ptr addrspace(3) %400, align 2, !tbaa !32
  %402 = insertelement <4 x half> %398, half %401, i64 2
  %403 = or disjoint i32 %394, 208
  %404 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %403
  %405 = load half, ptr addrspace(3) %404, align 2, !tbaa !32
  %406 = insertelement <4 x half> %402, half %405, i64 3
  %407 = or disjoint i32 %388, 16
  %408 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %407
  %409 = load half, ptr addrspace(3) %408, align 2, !tbaa !32
  %410 = insertelement <4 x half> poison, half %409, i64 0
  %411 = or disjoint i32 %394, 80
  %412 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %411
  %413 = load half, ptr addrspace(3) %412, align 2, !tbaa !32
  %414 = insertelement <4 x half> %410, half %413, i64 1
  %415 = or disjoint i32 %388, 128
  %416 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %415
  %417 = load half, ptr addrspace(3) %416, align 2, !tbaa !32
  %418 = insertelement <4 x half> %414, half %417, i64 2
  %419 = or disjoint i32 %394, 192
  %420 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %419
  %421 = load half, ptr addrspace(3) %420, align 2, !tbaa !32
  %422 = insertelement <4 x half> %418, half %421, i64 3
  %423 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %406, <4 x half> %384, <4 x float> zeroinitializer)
  %424 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %422, <4 x half> %384, <4 x float> zeroinitializer)
  %425 = shl nuw nsw i32 %204, 4
  %426 = and i32 %425, 16
  %427 = or disjoint i32 %345, %426
  %428 = or disjoint i32 %427, %159
  %429 = or disjoint i32 %428, %352
  %430 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %429
  %431 = load <4 x half>, ptr addrspace(3) %430, align 8
  %432 = or disjoint i32 %388, 1024
  %433 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %432
  %434 = load half, ptr addrspace(3) %433, align 2, !tbaa !32
  %435 = insertelement <4 x half> poison, half %434, i64 0
  %436 = or disjoint i32 %394, 1088
  %437 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %436
  %438 = load half, ptr addrspace(3) %437, align 2, !tbaa !32
  %439 = insertelement <4 x half> %435, half %438, i64 1
  %440 = or disjoint i32 %388, 1168
  %441 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %440
  %442 = load half, ptr addrspace(3) %441, align 2, !tbaa !32
  %443 = insertelement <4 x half> %439, half %442, i64 2
  %444 = or disjoint i32 %394, 1232
  %445 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %444
  %446 = load half, ptr addrspace(3) %445, align 2, !tbaa !32
  %447 = insertelement <4 x half> %443, half %446, i64 3
  %448 = or disjoint i32 %388, 1040
  %449 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %448
  %450 = load half, ptr addrspace(3) %449, align 2, !tbaa !32
  %451 = insertelement <4 x half> poison, half %450, i64 0
  %452 = or disjoint i32 %394, 1104
  %453 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %452
  %454 = load half, ptr addrspace(3) %453, align 2, !tbaa !32
  %455 = insertelement <4 x half> %451, half %454, i64 1
  %456 = or disjoint i32 %388, 1152
  %457 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %456
  %458 = load half, ptr addrspace(3) %457, align 2, !tbaa !32
  %459 = insertelement <4 x half> %455, half %458, i64 2
  %460 = or disjoint i32 %394, 1216
  %461 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 1024), i32 %460
  %462 = load half, ptr addrspace(3) %461, align 2, !tbaa !32
  %463 = insertelement <4 x half> %459, half %462, i64 3
  %464 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %447, <4 x half> %431, <4 x float> %423)
  %465 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %463, <4 x half> %431, <4 x float> %424)
  %466 = extractelement <4 x float> %465, i64 3
  %467 = fptrunc float %466 to half
  %468 = bitcast half %467 to i16
  %469 = extractelement <4 x float> %465, i64 2
  %470 = fptrunc float %469 to half
  %471 = bitcast half %470 to i16
  %472 = extractelement <4 x float> %465, i64 1
  %473 = fptrunc float %472 to half
  %474 = bitcast half %473 to i16
  %475 = extractelement <4 x float> %465, i64 0
  %476 = fptrunc float %475 to half
  %477 = bitcast half %476 to i16
  %478 = extractelement <4 x float> %464, i64 3
  %479 = fptrunc float %478 to half
  %480 = bitcast half %479 to i16
  %481 = extractelement <4 x float> %464, i64 2
  %482 = fptrunc float %481 to half
  %483 = bitcast half %482 to i16
  %484 = extractelement <4 x float> %464, i64 1
  %485 = fptrunc float %484 to half
  %486 = bitcast half %485 to i16
  %487 = extractelement <4 x float> %464, i64 0
  %488 = fptrunc float %487 to half
  %489 = bitcast half %488 to i16
  %490 = zext i16 %489 to i64
  %491 = zext i16 %486 to i64
  %492 = shl nuw nsw i64 %491, 16
  %493 = or disjoint i64 %492, %490
  %494 = zext i16 %483 to i64
  %495 = shl nuw nsw i64 %494, 32
  %496 = or disjoint i64 %493, %495
  %497 = zext i16 %480 to i64
  %498 = shl nuw i64 %497, 48
  %499 = or disjoint i64 %496, %498
  %500 = zext i16 %477 to i64
  %501 = zext i16 %474 to i64
  %502 = shl nuw nsw i64 %501, 16
  %503 = or disjoint i64 %502, %500
  %504 = zext i16 %471 to i64
  %505 = shl nuw nsw i64 %504, 32
  %506 = or disjoint i64 %503, %505
  %507 = zext i16 %468 to i64
  %508 = shl nuw i64 %507, 48
  %509 = or disjoint i64 %506, %508
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  br i1 %34, label %549, label %510

510:                                              ; preds = %119
  %511 = icmp slt i32 %368, 64
  %512 = icmp sgt i32 %367, -1
  %513 = and i1 %511, %512
  br i1 %513, label %514, label %524

514:                                              ; preds = %510
  %515 = getelementptr %struct.__half, ptr addrspace(4) %4, i64 %369
  %516 = getelementptr i8, ptr addrspace(4) %515, i64 %169
  %517 = load i32, ptr addrspace(4) %516, align 16, !tbaa !17
  %518 = getelementptr inbounds i8, ptr addrspace(4) %516, i64 4
  %519 = load i32, ptr addrspace(4) %518, align 4, !tbaa !17
  %520 = getelementptr inbounds i8, ptr addrspace(4) %516, i64 8
  %521 = load i32, ptr addrspace(4) %520, align 8, !tbaa !17
  %522 = getelementptr inbounds i8, ptr addrspace(4) %516, i64 12
  %523 = load i32, ptr addrspace(4) %522, align 4, !tbaa !17
  br label %524

524:                                              ; preds = %514, %510
  %525 = phi i32 [ %523, %514 ], [ 0, %510 ]
  %526 = phi i32 [ %521, %514 ], [ 0, %510 ]
  %527 = phi i32 [ %519, %514 ], [ 0, %510 ]
  %528 = phi i32 [ %517, %514 ], [ 0, %510 ]
  store i32 %528, ptr addrspace(3) %370, align 16, !tbaa !17
  store i32 %527, ptr addrspace(3) %371, align 4, !tbaa !17
  store i32 %526, ptr addrspace(3) %372, align 8, !tbaa !17
  store i32 %525, ptr addrspace(3) %373, align 4, !tbaa !17
  %529 = icmp slt i32 %368, 63
  %530 = add i32 %367, 16
  %531 = icmp sgt i32 %530, -1
  %532 = and i1 %529, %531
  br i1 %532, label %533, label %544

533:                                              ; preds = %524
  %534 = getelementptr %struct.__half, ptr addrspace(4) %4, i64 %369
  %535 = getelementptr i8, ptr addrspace(4) %534, i64 %169
  %536 = getelementptr i8, ptr addrspace(4) %535, i64 4096
  %537 = load i32, ptr addrspace(4) %536, align 16, !tbaa !17
  %538 = getelementptr i8, ptr addrspace(4) %535, i64 4100
  %539 = load i32, ptr addrspace(4) %538, align 4, !tbaa !17
  %540 = getelementptr i8, ptr addrspace(4) %535, i64 4104
  %541 = load i32, ptr addrspace(4) %540, align 8, !tbaa !17
  %542 = getelementptr i8, ptr addrspace(4) %535, i64 4108
  %543 = load i32, ptr addrspace(4) %542, align 4, !tbaa !17
  br label %544

544:                                              ; preds = %533, %524
  %545 = phi i32 [ %543, %533 ], [ 0, %524 ]
  %546 = phi i32 [ %541, %533 ], [ 0, %524 ]
  %547 = phi i32 [ %539, %533 ], [ 0, %524 ]
  %548 = phi i32 [ %537, %533 ], [ 0, %524 ]
  store i32 %548, ptr addrspace(3) %374, align 16, !tbaa !17
  store i32 %547, ptr addrspace(3) %375, align 4, !tbaa !17
  store i32 %546, ptr addrspace(3) %376, align 8, !tbaa !17
  store i32 %545, ptr addrspace(3) %377, align 4, !tbaa !17
  br label %559

549:                                              ; preds = %119
  %550 = shl nsw i32 %6, 17
  %551 = or disjoint i32 %550, %124
  %552 = or disjoint i32 %551, %122
  %553 = or disjoint i32 %552, 64
  %554 = shl nuw nsw i32 %12, 12
  %555 = add nuw nsw i32 %553, %554
  %556 = zext nneg i32 %555 to i64
  %557 = getelementptr inbounds %struct.__half, ptr addrspace(4) %4, i64 %556
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %370, ptr addrspace(4) noundef align 16 dereferenceable(16) %557, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !33
  %558 = getelementptr inbounds i8, ptr addrspace(4) %557, i64 4096
  tail call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %374, ptr addrspace(4) noundef align 16 dereferenceable(16) %558, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !33
  br label %559

559:                                              ; preds = %549, %544
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %560 = load <4 x half>, ptr addrspace(3) %383, align 8
  %561 = load half, ptr addrspace(3) %389, align 2, !tbaa !32
  %562 = insertelement <4 x half> poison, half %561, i64 0
  %563 = load half, ptr addrspace(3) %396, align 2, !tbaa !32
  %564 = insertelement <4 x half> %562, half %563, i64 1
  %565 = load half, ptr addrspace(3) %400, align 2, !tbaa !32
  %566 = insertelement <4 x half> %564, half %565, i64 2
  %567 = load half, ptr addrspace(3) %404, align 2, !tbaa !32
  %568 = insertelement <4 x half> %566, half %567, i64 3
  %569 = load half, ptr addrspace(3) %408, align 2, !tbaa !32
  %570 = insertelement <4 x half> poison, half %569, i64 0
  %571 = load half, ptr addrspace(3) %412, align 2, !tbaa !32
  %572 = insertelement <4 x half> %570, half %571, i64 1
  %573 = load half, ptr addrspace(3) %416, align 2, !tbaa !32
  %574 = insertelement <4 x half> %572, half %573, i64 2
  %575 = load half, ptr addrspace(3) %420, align 2, !tbaa !32
  %576 = insertelement <4 x half> %574, half %575, i64 3
  %577 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %568, <4 x half> %560, <4 x float> zeroinitializer)
  %578 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %576, <4 x half> %560, <4 x float> zeroinitializer)
  %579 = load <4 x half>, ptr addrspace(3) %430, align 8
  %580 = load half, ptr addrspace(3) %433, align 2, !tbaa !32
  %581 = insertelement <4 x half> poison, half %580, i64 0
  %582 = load half, ptr addrspace(3) %437, align 2, !tbaa !32
  %583 = insertelement <4 x half> %581, half %582, i64 1
  %584 = load half, ptr addrspace(3) %441, align 2, !tbaa !32
  %585 = insertelement <4 x half> %583, half %584, i64 2
  %586 = load half, ptr addrspace(3) %445, align 2, !tbaa !32
  %587 = insertelement <4 x half> %585, half %586, i64 3
  %588 = load half, ptr addrspace(3) %449, align 2, !tbaa !32
  %589 = insertelement <4 x half> poison, half %588, i64 0
  %590 = load half, ptr addrspace(3) %453, align 2, !tbaa !32
  %591 = insertelement <4 x half> %589, half %590, i64 1
  %592 = load half, ptr addrspace(3) %457, align 2, !tbaa !32
  %593 = insertelement <4 x half> %591, half %592, i64 2
  %594 = load half, ptr addrspace(3) %461, align 2, !tbaa !32
  %595 = insertelement <4 x half> %593, half %594, i64 3
  %596 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %587, <4 x half> %579, <4 x float> %577)
  %597 = tail call contract <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %595, <4 x half> %579, <4 x float> %578)
  %598 = extractelement <4 x float> %597, i64 3
  %599 = fptrunc float %598 to half
  %600 = extractelement <4 x float> %597, i64 2
  %601 = fptrunc float %600 to half
  %602 = extractelement <4 x float> %597, i64 1
  %603 = fptrunc float %602 to half
  %604 = extractelement <4 x float> %597, i64 0
  %605 = fptrunc float %604 to half
  %606 = extractelement <4 x float> %596, i64 3
  %607 = fptrunc float %606 to half
  %608 = extractelement <4 x float> %596, i64 2
  %609 = fptrunc float %608 to half
  %610 = extractelement <4 x float> %596, i64 1
  %611 = fptrunc float %610 to half
  %612 = extractelement <4 x float> %596, i64 0
  %613 = fptrunc float %612 to half
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %614 = shl nuw nsw i32 %16, 7
  %615 = and i32 %614, 1920
  %616 = and i32 %154, 480
  %617 = add nuw nsw i32 %615, %616
  %618 = and i32 %17, 12
  %619 = or disjoint i32 %617, %618
  %620 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %619
  store i64 %499, ptr addrspace(3) %620, align 8
  %621 = getelementptr inbounds i8, ptr addrspace(3) %620, i32 128
  store half %613, ptr addrspace(3) %621, align 8
  %622 = getelementptr inbounds i8, ptr addrspace(3) %620, i32 130
  store half %611, ptr addrspace(3) %622, align 2
  %623 = getelementptr inbounds i8, ptr addrspace(3) %620, i32 132
  store half %609, ptr addrspace(3) %623, align 4
  %624 = getelementptr inbounds i8, ptr addrspace(3) %620, i32 134
  store half %607, ptr addrspace(3) %624, align 2
  %625 = getelementptr inbounds %struct.__half, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @buf_dyn_shmem, i32 32), i32 %619
  store i64 %509, ptr addrspace(3) %625, align 8
  %626 = getelementptr inbounds i8, ptr addrspace(3) %625, i32 128
  store half %605, ptr addrspace(3) %626, align 8
  %627 = getelementptr inbounds i8, ptr addrspace(3) %625, i32 130
  store half %603, ptr addrspace(3) %627, align 2
  %628 = getelementptr inbounds i8, ptr addrspace(3) %625, i32 132
  store half %601, ptr addrspace(3) %628, align 4
  %629 = getelementptr inbounds i8, ptr addrspace(3) %625, i32 134
  store half %599, ptr addrspace(3) %629, align 2
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %630 = getelementptr inbounds %struct.__half, ptr addrspace(3) @buf_dyn_shmem, i32 %123
  %631 = add nuw nsw i32 %136, %123
  %632 = zext nneg i32 %631 to i64
  %633 = getelementptr inbounds %struct.__half, ptr addrspace(1) %2, i64 %632
  tail call void @llvm.memcpy.p1.p3.i64(ptr addrspace(1) noundef align 16 dereferenceable(16) %633, ptr addrspace(3) noundef align 16 dereferenceable(16) %630, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !34
  %634 = getelementptr inbounds i8, ptr addrspace(3) %630, i32 2048
  %635 = getelementptr inbounds i8, ptr addrspace(1) %633, i64 2048
  tail call void @llvm.memcpy.p1.p3.i64(ptr addrspace(1) noundef align 16 dereferenceable(16) %635, ptr addrspace(3) noundef align 16 dereferenceable(16) %634, i64 16, i1 false), !tbaa.struct !27, !call_argsrelate !34
  br label %636

636:                                              ; preds = %559, %5
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
!33 = !{i32 -1, i32 4, i32 -1, i32 -1}
!34 = !{i32 2, i32 -1, i32 -1, i32 -1}
