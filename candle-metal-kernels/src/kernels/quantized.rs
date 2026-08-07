use crate::utils::EncoderProvider;
use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};
use objc2_metal::MTLSize;

#[derive(Debug, Clone, Copy)]
pub enum GgmlDType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    F16,
    F32,
    BF16,
}

#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    // Byte offset of the weights inside `rhs`. Non-zero when the weight tensor is a view into a
    // larger allocation, e.g. one expert of a stacked MoE weight.
    rhs_offset: usize,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;

    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;

    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;

    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;

    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    let (nth0, nth1, align) = mv_threadgroup_shape(dtype);
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mv_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mv_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mv_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mv_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_q8_0_f32",
        GgmlDType::Q8_1 => "kernel_mul_mv_q8_1_f32",
        GgmlDType::Q2K => "kernel_mul_mv_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mv_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mv_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mv_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mv_q6_K_f32",
        GgmlDType::Q8K => "kernel_mul_mv_q8_K_f32",
        GgmlDType::F16 => "kernel_mul_mv_f16_f32",
        GgmlDType::BF16 => "kernel_mul_mv_bf16_f32",
        GgmlDType::F32 => "kernel_mul_mv_f32_f32",
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qmm_mv {name} B={b} M={m} K={k} N={n}");

    set_params!(
        encoder,
        (
            (rhs, rhs_offset),
            (lhs, lhs_offset),
            Output::with_offset(dst, dst_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Rows of `src0` a single threadgroup of `kernel_mul_mv_*_f32` (and hence of
/// `kernel_mul_mv_id_*_f32`) produces, together with the threadgroup shape that produces them.
/// These are properties of the kernel bodies, not tuning knobs: `kernel_mul_mv_q6_K_f32_impl`
/// writes `2*r0 + sgitg`, so two rows per threadgroup and nothing else will do.
fn mv_threadgroup_shape(dtype: GgmlDType) -> (usize, usize, usize) {
    match dtype {
        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_0
        | GgmlDType::Q8_1 => (8, 8, 8),
        // Fixing a bug in Metal for GGML
        // https://github.com/ggerganov/llama.cpp/blob/b8109bc0139f15a5b321909f47510b89dca47ffc/ggml-metal.m#L1576
        GgmlDType::Q2K => (2, 32, 4),
        GgmlDType::Q4K => (4, 8, 4),
        GgmlDType::Q3K | GgmlDType::Q5K => (2, 32, 4),
        GgmlDType::Q6K => (2, 32, 2),
        GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q8K | GgmlDType::F32 => (32, 1, 8),
    }
}

/// Fused MoE matrix-vector: one quantized matmul per (token, expert-slot), reading each slot's
/// weights straight out of the stacked `[n_experts, n, k]` tensor at `expert_id * nb02`.
///
/// This binds `kernel_mul_mv_id`, which is ggml's and which THIS FILE ALREADY CONTAINED --
/// `quantized.metal` is ggml-derived and carries the `_id` wrappers for q4_K, q5_K, q6_K and
/// q8_0 along with everything else. What was missing was only a Rust caller: nothing in candle
/// ever dispatched them, because `candle_nn::moe` routes quantized MoE to CUDA. So no kernel had
/// to be written, and none was: the shader below the Rust is untouched ggml.
///
/// It replaces the "split the stack into `n_experts` 2-D tensors, group the tokens on the host,
/// and issue one matmul per non-empty group" loop, which for a single decode token means eight
/// dispatches plus a routing readback that forces a GPU sync.
///
/// Shapes follow `QCudaStorage::indexed_moe_forward`, so the two backends present one contract:
/// - `src0` `[n_experts, n, k]`, quantized, expert stride `expert_stride_bytes`
/// - `src1` `[batch, in_dim1, k]` f32, `in_dim1` either 1 (gate/up) or `topk` (down)
/// - `ids`  `[batch, topk]` u32
/// - `dst`  `[batch, topk, n]` f32
///
/// `dtype` MUST be the quant type of the tensor actually being multiplied. The 35B's
/// `ffn_down_exps` is Q5_K on 37 layers and Q6_K on 3, so a caller that reads the type once and
/// reuses it silently mis-dequantizes three layers.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_id(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    (n_experts, n, k): (usize, usize, usize),
    (batch, in_dim1, topk): (usize, usize, usize),
    // Bytes between consecutive experts in `src0`. The caller computes it from the block layout
    // (`n * k / block_size * type_size`); it is not derivable here because this crate's
    // `GgmlDType` carries no block geometry.
    expert_stride_bytes: usize,
    src0: &Buffer,
    src0_offset: usize,
    src1: &Buffer,
    src1_offset: usize,
    ids: &Buffer,
    ids_offset: usize,
    dst: &Buffer,
    dst_offset: usize,
) -> Result<(), MetalKernelError> {
    let name = match dtype {
        GgmlDType::Q4K => "kernel_mul_mv_id_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mv_id_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mv_id_q6_K_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_id_q8_0_f32",
        GgmlDType::Q4_0 => "kernel_mul_mv_id_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mv_id_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mv_id_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mv_id_q5_1_f32",
        GgmlDType::Q3K => "kernel_mul_mv_id_q3_K_f32",
        // Q2K is DELIBERATELY REFUSED even though kernel_mul_mv_id_q2_K_f32 exists.
        // kernel_mul_mv_q2_K_f32_impl writes `(r0*N_SIMDGROUP + sgitg)*N_DST` = EIGHT rows per
        // threadgroup (quantized.metal:4596) and, unlike the legacy quants at :2374 and :2551, it
        // has no `first_row + row < ne01` guard on the store. mv_threadgroup_shape reports 4 for
        // Q2K, so the dispatch launches twice the threadgroups needed and each one writes eight
        // rows: for a plain mv_t the overrun lands past the end of a single output row (a latent
        // out-of-bounds write that predates this function), but for mv_id the next expert slot is
        // RIGHT THERE in the same dst buffer and gets clobbered. Measured: 2 of 6 slots wrong.
        // Fixing it means correcting Q2K's entry in the shared threadgroup table, which changes
        // call_quantized_matmul_mv_t for every existing caller -- out of scope here.
        GgmlDType::Q2K => Err(MetalKernelError::UnsupportedDTypeForOp("Q2K", "mul_mv_id"))?,
        // F16 and F32 are DELIBERATELY REFUSED even though kernel_mul_mv_id_{f16,f32}_f32 exist.
        // They are the only instantiations that go through the nb-forwarding `mmv_fn` overload
        // (quantized.metal:7489) into `kernel_mul_mv_impl`, and that kernel is shaped differently
        // from every quantized one: `offset0 = r0*nb01` makes tgpig.x ONE output row, and
        // `rb = tgpig.y*N_MV_T_T` makes tgpig.y a block of four src1 columns. So it needs both a
        // real nb01 and a different grid (width ne01, height ne11/4) than
        // `mv_threadgroup_shape` describes. Dispatching it on the quantized geometry with nb01=0
        // would make every threadgroup read row 0 and silently return garbage.
        //
        // (call_quantized_matmul_mv_t has the same mismatch for these two types today. That is a
        // pre-existing candle bug, not one introduced here, and fixing it is out of scope -- but
        // it is why "just pass nb01" is not the fix for this function either.)
        GgmlDType::F16 => Err(MetalKernelError::UnsupportedDTypeForOp("F16", "mul_mv_id"))?,
        GgmlDType::F32 => Err(MetalKernelError::UnsupportedDTypeForOp("F32", "mul_mv_id"))?,
        GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "mul_mv_id"))?,
        GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "mul_mv_id"))?,
        GgmlDType::BF16 => Err(MetalKernelError::UnsupportedDTypeForOp("BF16", "mul_mv_id"))?,
    };
    if n_experts == 0 {
        return Err(MetalKernelError::InvalidInput(
            "mul_mv_id: expert stack is empty".to_string(),
        ));
    }
    if in_dim1 != 1 && in_dim1 != topk {
        return Err(MetalKernelError::InvalidInput(format!(
            "mul_mv_id: src1 dim1 {in_dim1} must be 1 or topk {topk}"
        )));
    }
    let (nth0, nth1, align) = mv_threadgroup_shape(dtype);
    if !n.is_multiple_of(align) {
        return Err(MetalKernelError::InvalidInput(format!(
            "mul_mv_id {name}: output dim {n} is not a multiple of the kernel's {align} rows per \
             threadgroup; the kernel writes past the row it was asked for"
        )));
    }

    let f32_size = core::mem::size_of::<f32>();
    let ne00 = k as i64;
    let ne01 = n as i64;
    let nb02 = expert_stride_bytes as u64;
    // src1 is contiguous [batch, in_dim1, k] f32. nb11 walks a slot, nb12 walks a token.
    let nb10 = f32_size as u64;
    let nb11 = (k * f32_size) as u64;
    let nb12 = (in_dim1 * k * f32_size) as u64;
    // ne1 is what the kernel multiplies iid1 by when it rebases dst, so it must be topk: dst is
    // [batch, topk, n] and dst_cur = dst + idx*ne0 + iid1*ne1*ne0.
    let ne1 = topk as i64;

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "qmm_mv_id {name} experts={n_experts} N={n} K={k} batch={batch} topk={topk}"
    );

    set_params!(
        encoder,
        (
            (src0, src0_offset),
            (src1, src1_offset),
            Output::with_offset(dst, dst_offset),
            (ids, ids_offset),
            topk as i64,                                 // nei0
            batch as i64,                                // nei1
            (topk * core::mem::size_of::<u32>()) as u64, // nbi1
            ne00,
            ne01,
            1i64, // ne02, forced to 1 inside the kernel
            // nb00/nb01 are dropped on the floor by the `mmv_fn` overload every dtype reaching
            // here goes through (quantized.metal:7511): it forwards only ne*/r2/r3 to the
            // impl. The one overload that DOES forward them serves f16/f32, which are refused
            // above -- so 0 here is not "unused by k-quants", it is never read at all.
            0u64,
            0u64,
            nb02,
            ne00,           // ne10
            in_dim1 as i64, // ne11
            1i64,           // ne12
            1i64,           // ne13
            nb10,
            nb11,
            nb12,
            ne01, // ne0
            ne1,
            0u64 // nb1, unused by the k-quant impls
        )
    );

    // kernel_mul_mv_id declares `threadgroup int8_t * shared_values [[threadgroup(0)]]`. The
    // k-quant impls it is instantiated with never read it, but Metal still needs a length bound
    // for the binding; 8192 is what ggml allocates here.
    encoder.set_threadgroup_memory_length(0, 8192);

    encoder.dispatch_thread_groups(
        MTLSize {
            width: divide(n, align),
            height: 1,
            depth: batch * topk,
        },
        MTLSize {
            width: nth0,
            height: nth1,
            depth: 1,
        },
    );
    Ok(())
}

/// - src0 is usually weight
/// - src1 is usually xs
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    src0_shape: &[usize],
    src0_stride: &[usize],
    src0: &Buffer,
    // Byte offset of the weights inside `src0`; see `call_quantized_matmul_mv_t`.
    src0_offset: usize,
    src1_shape: &[usize],
    src1_stride: &[usize],
    src1: &Buffer,
    src1_offset: usize,
    dst_shape: &[usize],
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = src0_shape[src0_shape.len() - 1] as i64;
    let ne01 = src0_shape[src0_shape.len() - 2] as i64;
    let ne02 = src0_shape[src0_shape.len() - 3] as i64;
    let ne03 = src0_shape[src0_shape.len() - 4] as i64;

    let nb01 = src0_stride[src0_stride.len() - 2] as i64;
    let nb02 = src0_stride[src0_stride.len() - 3] as i64;
    let nb03 = src0_stride[src0_stride.len() - 4] as i64;

    let ne11 = src1_shape[src1_shape.len() - 2] as i64;
    let ne12 = src1_shape[src1_shape.len() - 3] as i64;
    let ne13 = src1_shape[src1_shape.len() - 4] as i64;

    let nb10 = src1_stride[src1_stride.len() - 1] as i64;
    let nb11 = src1_stride[src1_stride.len() - 2] as i64;
    let nb12 = src1_stride[src1_stride.len() - 3] as i64;
    let nb13 = src1_stride[src1_stride.len() - 4] as i64;

    let ne0 = dst_shape[dst_shape.len() - 1] as i64;
    let ne1 = dst_shape[dst_shape.len() - 2] as i64;
    let r2 = (ne12 / ne02) as u32;
    let r3 = (ne13 / ne03) as u32;

    let thread_groups_count = MTLSize {
        width: divide(ne11 as usize, 32),
        height: divide(ne01 as usize, 64),
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mm_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mm_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mm_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mm_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mm_q8_0_f32",
        GgmlDType::Q2K => "kernel_mul_mm_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mm_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mm_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mm_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mm_q6_K_f32",
        GgmlDType::F16 => "kernel_mul_mm_f16_f32",
        GgmlDType::BF16 => "kernel_mul_mm_bf16_f32",
        GgmlDType::F32 => "kernel_mul_mm_f32_f32",
        GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "qmatmul"))?,
        GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "qmatmul"))?,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qmm_mm {name} M={ne11} K={ne00} N={ne01}");

    set_params!(
        encoder,
        (
            (src0, src0_offset),
            (src1, src1_offset),
            Output::with_offset(dst, dst_offset),
            ne00,
            ne02,
            nb01,
            nb02,
            nb03,
            ne12,
            nb10,
            nb11,
            nb12,
            nb13,
            ne0,
            ne1,
            r2,
            r3
        )
    );

    encoder.set_threadgroup_memory_length(0, 8192);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_quantized_get_rows(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    hidden_size: usize,
    row_stride: usize,
    ids_len: usize,
    src: &Buffer,
    ids: &Buffer,
    ids_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let dst_row_stride = hidden_size * core::mem::size_of::<f32>();
    let name = match dtype {
        GgmlDType::F32 => "kernel_get_rows_f32",
        GgmlDType::F16 => "kernel_get_rows_f16",
        GgmlDType::BF16 => "kernel_get_rows_bf16",
        GgmlDType::Q4_0 => "kernel_get_rows_q4_0",
        GgmlDType::Q4_1 => "kernel_get_rows_q4_1",
        GgmlDType::Q5_0 => "kernel_get_rows_q5_0",
        GgmlDType::Q5_1 => "kernel_get_rows_q5_1",
        GgmlDType::Q8_0 => "kernel_get_rows_q8_0",
        GgmlDType::Q2K => "kernel_get_rows_q2_K",
        GgmlDType::Q3K => "kernel_get_rows_q3_K",
        GgmlDType::Q4K => "kernel_get_rows_q4_K",
        GgmlDType::Q5K => "kernel_get_rows_q5_K",
        GgmlDType::Q6K => "kernel_get_rows_q6_K",
        GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "get_rows"))?,
        GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "get_rows"))?,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "qget_rows {name} ids={ids_len} hidden={hidden_size}"
    );

    let thread_groups_count = MTLSize {
        width: ids_len,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };

    set_params!(
        encoder,
        (
            src,
            (ids, ids_offset),
            Output::new(dst),
            hidden_size as i64,
            row_stride as u64,
            0u64,
            ids_len as i64,
            core::mem::size_of::<u32>() as u64,
            0u64,
            dst_row_stride as u64,
            0u64
        )
    );

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

fn divide(m: usize, b: usize) -> usize {
    m.div_ceil(b)
}
