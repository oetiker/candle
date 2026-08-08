use crate::utils::EncoderProvider;
use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};
use objc2_metal::MTLSize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Stable short label for error messages, kept as an explicit match (not `Debug`) so the exact
/// strings `mul_mv_id`/`mul_mm_id` error messages have always used stay stable across refactors.
fn dtype_label(dtype: GgmlDType) -> &'static str {
    match dtype {
        GgmlDType::Q4_0 => "Q4_0",
        GgmlDType::Q4_1 => "Q4_1",
        GgmlDType::Q5_0 => "Q5_0",
        GgmlDType::Q5_1 => "Q5_1",
        GgmlDType::Q8_0 => "Q8_0",
        GgmlDType::Q8_1 => "Q8_1",
        GgmlDType::Q2K => "Q2K",
        GgmlDType::Q3K => "Q3K",
        GgmlDType::Q4K => "Q4K",
        GgmlDType::Q5K => "Q5K",
        GgmlDType::Q6K => "Q6K",
        GgmlDType::Q8K => "Q8K",
        GgmlDType::F16 => "F16",
        GgmlDType::F32 => "F32",
        GgmlDType::BF16 => "BF16",
    }
}

/// Look `dtype` up in `allowed` (an explicit `(dtype, kernel_name)` allow-list) or refuse it for
/// `op`. Shared by `mul_mv_id` and `mul_mm_id` so their dtype dispatch is the same
/// lookup-or-refuse shape instead of two independently hand-typed matches over all 15
/// `GgmlDType` variants -- the two functions' allow-lists genuinely differ (`mul_mv_id`
/// additionally supports the legacy quants and Q3K; `mul_mm_id` is restricted to the four
/// `SUPPORTED_EXPERT_QUANTS` types), so this takes the list as a parameter rather than hardcoding
/// one.
fn require_supported_dtype(
    dtype: GgmlDType,
    allowed: &[(GgmlDType, &'static str)],
    op: &'static str,
) -> Result<&'static str, MetalKernelError> {
    allowed
        .iter()
        .find(|(d, _)| *d == dtype)
        .map(|(_, name)| *name)
        .ok_or_else(|| MetalKernelError::UnsupportedDTypeForOp(dtype_label(dtype), op))
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
    // Q2K, F16, F32, Q8_1, Q8K, BF16 are DELIBERATELY excluded from this allow-list, even though
    // `kernel_mul_mv_id_{q2_K,f16,f32}_f32` exist:
    //
    // - Q2K: `kernel_mul_mv_q2_K_f32_impl` writes `(r0*N_SIMDGROUP + sgitg)*N_DST` = EIGHT rows
    //   per threadgroup (quantized.metal:4596) and, unlike the legacy quants at :2374 and :2551,
    //   has no `first_row + row < ne01` guard on the store. `mv_threadgroup_shape` reports 4 for
    //   Q2K, so the dispatch launches twice the threadgroups needed and each one writes eight
    //   rows: for a plain mv_t the overrun lands past the end of a single output row (a latent
    //   out-of-bounds write that predates this function), but for mv_id the next expert slot is
    //   RIGHT THERE in the same dst buffer and gets clobbered. Measured: 2 of 6 slots wrong.
    //   Fixing it means correcting Q2K's entry in the shared threadgroup table, which changes
    //   `call_quantized_matmul_mv_t` for every existing caller -- out of scope here.
    // - F16, F32: the only instantiations that go through the nb-forwarding `mmv_fn` overload
    //   (quantized.metal:7489) into `kernel_mul_mv_impl`, shaped differently from every quantized
    //   one: `offset0 = r0*nb01` makes tgpig.x ONE output row, and `rb = tgpig.y*N_MV_T_T` makes
    //   tgpig.y a block of four src1 columns. So it needs both a real nb01 and a different grid
    //   (width ne01, height ne11/4) than `mv_threadgroup_shape` describes. Dispatching it on the
    //   quantized geometry with nb01=0 would make every threadgroup read row 0 and silently
    //   return garbage. (`call_quantized_matmul_mv_t` has the same mismatch for these two types
    //   today -- a pre-existing candle bug, not one introduced here, and out of scope to fix, but
    //   why "just pass nb01" is not the fix for this function either.)
    // - Q8_1, Q8K: no `_id` kernel instantiation exists for these at all.
    const MUL_MV_ID_SUPPORTED: &[(GgmlDType, &str)] = &[
        (GgmlDType::Q4K, "kernel_mul_mv_id_q4_K_f32"),
        (GgmlDType::Q5K, "kernel_mul_mv_id_q5_K_f32"),
        (GgmlDType::Q6K, "kernel_mul_mv_id_q6_K_f32"),
        (GgmlDType::Q8_0, "kernel_mul_mv_id_q8_0_f32"),
        (GgmlDType::Q4_0, "kernel_mul_mv_id_q4_0_f32"),
        (GgmlDType::Q4_1, "kernel_mul_mv_id_q4_1_f32"),
        (GgmlDType::Q5_0, "kernel_mul_mv_id_q5_0_f32"),
        (GgmlDType::Q5_1, "kernel_mul_mv_id_q5_1_f32"),
        (GgmlDType::Q3K, "kernel_mul_mv_id_q3_K_f32"),
    ];
    let name = require_supported_dtype(dtype, MUL_MV_ID_SUPPORTED, "mul_mv_id")?;
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

/// The largest `topk * batch` (ggml's `dst_rows`) a single `mul_mm_id` dispatch can hold in
/// `device`'s threadgroup memory. SINGLE SOURCE OF TRUTH for this formula: it must be called from
/// both [`call_quantized_matmul_mm_id`]'s hard-reject check below AND from the caller that sizes
/// chunks against it (`QMetalStorage::indexed_moe_forward` in candle-core) -- this used to be the
/// same expression typed out at both sites, and a silent drift between them (say, one site's
/// constant changing without the other's) would have let the caller size a chunk the callee then
/// rejects, or worse, would have needed BOTH sites to indepedently stay correct for a hard
/// hardware ceiling never to be exceeded.
///
/// Ported directly from ggml's own host-side gate for this kernel (`ggml-metal.m` circa candle's
/// vendor point `611aa914e^`: `dst_rows_max = (device.maxThreadgroupMemoryLength/2 - 8192)/4`),
/// not independently derived. Of its three constants, two are read straight off the kernel body:
/// `8192` is `kernel_mul_mm_id`'s fixed simdgroup-matrix staging area in bytes, and `4` is
/// `sizeof(ushort2)`, the size of one `rowids` entry. The leading `/ 2`, however, is NOT explained
/// anywhere in ggml's source or derived by this port -- it is an unexplained safety margin,
/// inherited as-is. It only makes the bound MORE conservative (a smaller `dst_rows_max` can cause
/// this function to refuse a batch that would in fact have fit in threadgroup memory, never
/// accept one that overflows it), so carrying it forward un-investigated is safe, just not fully
/// understood.
pub fn mul_mm_id_dst_rows_max(device: &Device) -> usize {
    use objc2_metal::MTLDevice as _;
    let max_tg_mem = device.as_ref().maxThreadgroupMemoryLength();
    (max_tg_mem / 2).saturating_sub(8192) / 4
}

/// Fused MoE matrix-MATRIX (prefill): ggml's `kernel_mul_mm_id`, a tiled simdgroup-matrix kernel
/// that -- unlike `mul_mv_id`'s one-threadgroup-per-output-row scheme -- amortizes each 64-row
/// tile of an expert's weights over up to 32 (token, slot) pairs at once. Grid depth is
/// `n_experts`: EVERY expert is dispatched, and each expert's threadgroup scans the full `ids`
/// array (the "TODO: parallelize this loop" in `quantized.metal`'s `kernel_mul_mm_id`) to collect
/// just the (token, slot) pairs routed to it, into a threadgroup-memory `rowids` scratch array.
///
/// This crate's `quantized.metal` is ggml-derived and already contains the whole
/// `kernel_mul_mm_id_*` family (verified against `kernel_mul_mv_id`'s sibling story) -- nothing
/// here is a new kernel, only a new Rust caller for one that already existed unreferenced.
///
/// Same shape contract as [`call_quantized_matmul_mv_id`]:
/// - `src0` `[n_experts, n, k]`, quantized, row stride `row_stride_bytes`, expert stride
///   `expert_stride_bytes`
/// - `src1` `[batch, in_dim1, k]` f32, `in_dim1` either 1 (gate/up) or `topk` (down)
/// - `ids`  `[batch, topk]` u32
/// - `dst`  `[batch, topk, n]` f32
///
/// # The threadgroup-memory ceiling this function enforces, and why it cannot just chunk itself
///
/// The kernel keeps a `rowids` scratch array in threadgroup memory sized for the worst case of
/// `topk * batch` entries (ggml's `dst_rows`; an expert cannot appear twice in one token's
/// top-k list, so `batch` alone bounds how many rows any single expert can collect, and `topk`
/// is the per-token stride ggml's own host code multiplies by -- see `ggml-metal.m` circa
/// `611aa914e^`, `dst_rows_max`). `dst_rows*4` bytes of scratch sit past an 8192-byte
/// simdgroup-matrix staging area, and Metal's `maxThreadgroupMemoryLength` is a hard per-dispatch
/// ceiling (32768 bytes measured on M1 Max) -- there is no larger tier to request, and exceeding
/// it is not a slowdown, it is a pipeline-creation/dispatch failure. A 5304-token prefill at
/// `topk=8` is `dst_rows=42432`, forty times the ~2048-row budget that ceiling leaves: dispatching
/// the whole prompt as one call is not an option on this hardware. So this function REFUSES any
/// `(batch, topk)` whose `dst_rows` does not fit, rather than silently truncating `rowids` and
/// losing token rows. The caller (`QMetalStorage::indexed_moe_forward`) chunks the token axis
/// into sub-batches that fit and issues one dispatch per chunk, mirroring how ggml itself never
/// sees more than one `n_ubatch` worth of tokens in a single `mul_mat_id` call.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_id(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    (n_experts, n, k): (usize, usize, usize),
    (batch, in_dim1, topk): (usize, usize, usize),
    // Bytes between consecutive OUTPUT ROWS within one expert's matrix (nb01), and between
    // consecutive experts (nb02, same quantity `call_quantized_matmul_mv_id` takes). Both are
    // block-layout facts this crate's dtype-less `GgmlDType` cannot derive; the caller computes
    // them from `n`, `k` and the *candle-core* `GgmlDType`'s block size / type size.
    row_stride_bytes: usize,
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
    // Every dtype other than these four is refused here even though `kernel_mul_mm_id_*` exists
    // for most of them (see the f32/f16/bf16/legacy-quant/iq* template list in quantized.metal).
    //
    // UNLIKE `mul_mv_id`'s Q2K refusal above, this restriction is PRECAUTIONARY, not
    // load-bearing: confirmed by reading the shader, every `kernel_mul_mm_id_*` instantiation --
    // for every dtype, not just these four -- routes through the SAME shared
    // `kernel_mul_mm_id_impl` (quantized.metal:7145), templated only on block type and
    // dequantize function, with global `BLOCK_SIZE_M/N/K` `#define`s (quantized.metal:6984-6991)
    // it cannot diverge from per-dtype. That is structurally different from `mul_mv_id`, where
    // each dtype has its own kernel body and Q2K's row-writing genuinely diverged from what the
    // Rust-side `mv_threadgroup_shape` table promised. So the Q2K-style store-guard hazard class
    // CANNOT apply to `mul_mm_id` for any dtype -- a future reader should not assume it does.
    // This match exists purely because the model loader's `SUPPORTED_EXPERT_QUANTS` already
    // promises only these four reach here, so keeping this in sync is defense in depth, not a
    // live safety decision the way the `mul_mv_id` list above is.
    const MUL_MM_ID_SUPPORTED: &[(GgmlDType, &str)] = &[
        (GgmlDType::Q4K, "kernel_mul_mm_id_q4_K_f32"),
        (GgmlDType::Q5K, "kernel_mul_mm_id_q5_K_f32"),
        (GgmlDType::Q6K, "kernel_mul_mm_id_q6_K_f32"),
        (GgmlDType::Q8_0, "kernel_mul_mm_id_q8_0_f32"),
    ];
    let name = require_supported_dtype(dtype, MUL_MM_ID_SUPPORTED, "mul_mm_id")?;
    if n_experts == 0 {
        return Err(MetalKernelError::InvalidInput(
            "mul_mm_id: expert stack is empty".to_string(),
        ));
    }
    if in_dim1 != 1 && in_dim1 != topk {
        return Err(MetalKernelError::InvalidInput(format!(
            "mul_mm_id: src1 dim1 {in_dim1} must be 1 or topk {topk}"
        )));
    }
    // kernel_mul_mm_id_impl walks K in BLOCK_SIZE_K=32 strides with no tail handling; a K that
    // does not divide evenly would read the dequantized tile past what the caller supplied.
    if !k.is_multiple_of(32) {
        return Err(MetalKernelError::InvalidInput(format!(
            "mul_mm_id {name}: K {k} is not a multiple of 32, the kernel's BLOCK_SIZE_K"
        )));
    }

    let dst_rows = topk * batch;
    // SAFETY/CORRECTNESS: not a tuning knob. See `mul_mm_id_dst_rows_max`'s doc comment.
    let dst_rows_max = mul_mm_id_dst_rows_max(device);
    if dst_rows > dst_rows_max {
        let max_tg_mem = {
            use objc2_metal::MTLDevice as _;
            device.as_ref().maxThreadgroupMemoryLength()
        };
        return Err(MetalKernelError::InvalidInput(format!(
            "mul_mm_id {name}: topk*batch {dst_rows} exceeds this device's dst_rows_max \
             {dst_rows_max} (maxThreadgroupMemoryLength {max_tg_mem} bytes); caller must chunk \
             the token axis, one dispatch cannot hold this many rowids in threadgroup memory"
        )));
    }

    let f32_size = core::mem::size_of::<f32>();
    let nei0 = topk as i64;
    let nei1 = batch as i64;
    let nbi1 = (topk * core::mem::size_of::<u32>()) as u64;
    let ne00 = k as i64;
    let ne02 = n_experts as i64;
    let nb01 = row_stride_bytes as u64;
    let nb02 = expert_stride_bytes as u64;
    // src1 is contiguous [batch, in_dim1, k] f32: nb11 walks a slot, nb12 walks a token. Same
    // layout `call_quantized_matmul_mv_id` assumes.
    let nb10 = f32_size as u64;
    let nb11 = (k * f32_size) as u64;
    let nb12 = (in_dim1 * k * f32_size) as u64;
    let ne11 = in_dim1 as i64;
    let ne12 = batch as i64;
    let ne13 = 1i64;
    let ne0 = n as i64;
    // What the kernel multiplies a token's batch index by when it rebases dst: dst is
    // [batch, topk, n] and dst_cur = dst + jid[0]*ne0 + jid[1]*ne0*ne1, jid[0] the topk slot,
    // jid[1] the token. So ne1 here is topk, exactly as in mv_id.
    let ne1 = topk as i64;

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "qmm_mm_id {name} experts={n_experts} N={n} K={k} batch={batch} topk={topk}"
    );

    set_params!(
        encoder,
        (
            (src0, src0_offset),
            (src1, src1_offset),
            Output::with_offset(dst, dst_offset),
            (ids, ids_offset),
            nei0,
            nei1,
            nbi1,
            ne00,
            ne02,
            nb01,
            nb02,
            ne11,
            ne12,
            ne13,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            0u64 // nb1: declared by kernel_mul_mm_id (`constant uint64_t & nb1`) but never read
                 // inside kernel_mul_mm_id_impl -- dst indexing is computed from ne0/ne1/jid
                 // directly, the same way mv_id's trailing nb1 is dead.
        )
    );

    // rowids lives in threadgroup memory past the 8192-byte simdgroup-matrix staging area ggml
    // always reserves (matching the fixed 8192 call_quantized_matmul_mm_t already sets); the
    // dst_rows check above guarantees this fits under maxThreadgroupMemoryLength.
    encoder.set_threadgroup_memory_length(0, pad16(8192 + dst_rows * 4));

    encoder.dispatch_thread_groups(
        MTLSize {
            width: divide(batch, 32), // BLOCK_SIZE_N columns of the compacted (token,slot) list
            height: divide(n, 64),    // BLOCK_SIZE_M rows of the expert's output
            depth: n_experts,         // every expert gets a threadgroup; most collect zero rows
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn pad16(x: usize) -> usize {
    x.div_ceil(16) * 16
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
