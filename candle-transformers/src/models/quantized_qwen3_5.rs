//! Qwen3.5 implementation with quantization support.
//!
//! Qwen3.5 is a hybrid architecture combining Gated DeltaNet (Linear Attention)
//! and Gated Attention (Full Softmax Attention).
//!
//! Based on the Qwen 3.5 architecture and implemented with quantized weights
//! for reduced memory usage and faster inference.
//!
use super::with_tracing::QMatMul;
use crate::models::qwen3_5_linear_attn_scan::{gated_delta_rule_chunked, sequential_step};
use crate::{quantized_nn::RmsNorm, utils::repeat_kv};
use candle::quantized::{gguf_file, GgmlDType, QStorage, QTensor};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{kv_cache::ConcatKvCache, Activation, Embedding};
use std::borrow::Cow;
use std::io::{Read, Seek};
use std::sync::Arc;

pub struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    pub fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    pub fn qmatmul(&mut self, name: &str) -> Result<(QMatMul, usize)> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        let out_dim = ws.shape().dims()[0];
        Ok((QMatMul::from_weights(ws.into())?, out_dim))
    }

    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
}

#[derive(Debug, Clone)]
struct MlpWeights {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
}

impl MlpWeights {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let (gate_proj, _) = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let (up_proj, _) = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let (down_proj, _) = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        let act_fn = Activation::Silu;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        })
    }
}

impl Module for MlpWeights {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(x)?;
        let gated = (gate * up)?;
        self.down_proj.forward(&gated)
    }
}

/// Per-layer quant types for the three routed-expert weight stacks (`ffn_gate_exps`,
/// `ffn_up_exps`, `ffn_down_exps`). unsloth's dynamic quant on the 35B does NOT hold a single
/// type across layers -- `ffn_down_exps` is Q5_K on 37 layers and Q6_K on 3 -- so this must be
/// read per layer and carried alongside each layer's weights, never assumed from layer 0.
/// Tasks 3 and 4 dispatch the fused Metal MoE kernel on this, per layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertQuants {
    pub gate: GgmlDType,
    pub up: GgmlDType,
    pub down: GgmlDType,
}

/// Quant types this loader knows how to handle for routed-expert weights. Anything else fails
/// loudly at load time instead of silently mis-dequantizing.
const SUPPORTED_EXPERT_QUANTS: [GgmlDType; 4] = [
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
    GgmlDType::Q8_0,
];

fn check_expert_quant(tensor_name: &str, dtype: GgmlDType) -> Result<GgmlDType> {
    if SUPPORTED_EXPERT_QUANTS.contains(&dtype) {
        Ok(dtype)
    } else {
        candle::bail!(
            "unsupported expert quant type {dtype:?} on tensor {tensor_name}; expected one of \
             {SUPPORTED_EXPERT_QUANTS:?}"
        )
    }
}

/// Read `{gate,up,down}` quant types for every MoE layer straight from a GGUF file's tensor
/// table -- no dequantization, no full-tensor read, just the header. Exists so a loader that
/// collapses per-layer quant types to a single one (e.g. layer 0's) can be caught in a test
/// before the fused Metal kernel it feeds is written.
pub fn read_expert_quants_per_layer<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<Vec<ExpertQuants>> {
    let mut file = std::fs::File::open(path.as_ref())?;
    let ct = gguf_file::Content::read(&mut file)?;
    let arch = match ct.metadata.get("general.architecture") {
        Some(v) => v.to_string()?,
        None => candle::bail!("cannot find general.architecture in metadata"),
    };
    let md_get = |s: &str| {
        let keyed = s.replace("qwen3.", &format!("{arch}."));
        match ct.metadata.get(s).or_else(|| ct.metadata.get(&keyed)) {
            Some(v) => Ok(v),
            None => candle::bail!("cannot find {s} or {keyed} in metadata"),
        }
    };
    let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;

    let mut out = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let tensor_dtype = |suffix: &str| -> Result<GgmlDType> {
            let name = format!("blk.{i}.{suffix}");
            match ct.tensor_infos.get(&name) {
                Some(info) => check_expert_quant(&name, info.ggml_dtype),
                None => candle::bail!("missing tensor {name} in gguf tensor table"),
            }
        };
        out.push(ExpertQuants {
            gate: tensor_dtype("ffn_gate_exps.weight")?,
            up: tensor_dtype("ffn_up_exps.weight")?,
            down: tensor_dtype("ffn_down_exps.weight")?,
        });
    }
    Ok(out)
}

/// How the routed experts are evaluated, for ONE phase (prefill or decode -- see [`MoeModes`]).
///
/// Chosen at load time, not per call, because it decides what gets built: `Fused` needs the
/// stacked `[n_experts, n_out, n_in]` tensor the fused kernels index into, `Loop` needs only the
/// per-expert 2-D matrices. Both are always available once loaded (the per-expert matrices are
/// views into the stack, see [`split_stacked_experts`]), so the switch costs nothing but keeps
/// `Loop` byte-for-byte the path the llama.cpp gate was closed against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MoeMode {
    /// Group tokens by expert on the host and issue one plain quantized matmul per non-empty
    /// group. Correct everywhere; the reference path the llama.cpp gate was closed against.
    #[default]
    Loop,
    /// A fused Metal kernel: no host readback of the routing, no per-expert dispatch. Decode
    /// (`batch == 1`) uses `mul_mv_id`; prefill (`batch > 1`) uses `mul_mm_id`, chunked. Measured
    /// (Task 4, 35B/256 experts, 5304-token prompt): `mul_mv_id` at decode is a clear win
    /// (3.04 -> 11.77 tok/s over `Loop`, Task 3), but `mul_mm_id` at prefill is a REGRESSION
    /// versus `Loop` (78.0 vs 118.3 tok/s measured on the identical prompt) -- ggml's own
    /// `kernel_mul_mm_id` pays an unparallelized O(chunk^2 * n_experts) row-id scan that `Loop`'s
    /// one-matmul-per-expert amortizes away on a long prompt. So `Fused` is not a strict win per
    /// phase; [`MoeModes`] exists so a caller can pick `Fused` for decode and `Loop` for prefill
    /// independently, which is the configuration this measurement recommends.
    Fused,
}

impl std::str::FromStr for MoeMode {
    type Err = candle::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "loop" => Ok(Self::Loop),
            "fused" => Ok(Self::Fused),
            other => candle::bail!("unknown MoE mode {other}, expected `loop` or `fused`"),
        }
    }
}

/// [`MoeMode`], independently, for prefill (`batch > 1`) and decode (`batch == 1`).
///
/// A LOAD-time choice, like `MoeMode` itself: the stacked expert tensors `Fused` needs are built
/// once if EITHER phase asks for `Fused`, and both phases share them (the per-expert `QMatMul`s
/// are views into the same stack regardless of which phase, if any, uses `Fused`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MoeModes {
    pub prefill: MoeMode,
    pub decode: MoeMode,
}

impl MoeModes {
    /// Both phases the same mode -- what a bare `--moe loop|fused` means.
    pub fn both(mode: MoeMode) -> Self {
        Self {
            prefill: mode,
            decode: mode,
        }
    }

    fn any_fused(self) -> bool {
        self.prefill == MoeMode::Fused || self.decode == MoeMode::Fused
    }
}

/// Split a 3D stacked expert tensor `[n_experts, n_out, n_in]` into one `QMatMul` per expert.
///
/// candle's quantized MoE path is CUDA-only: `candle_nn::moe::moe_gemm_gguf` bails on every other
/// backend, and it works by handing `QTensor::device_ptr()` to a C kernel -- a method that itself
/// bails for Metal and CPU. So on Metal a 3D quantized stack cannot be fed to a plain matmul.
///
/// Each expert occupies a contiguous byte range (verified against the checkpoint: `ffn_gate_exps`
/// is 150994944 bytes for 256 experts, and 512*2048/256 blocks * 144 bytes = 589824 per expert), so
/// the stack can be carved into 2D `[n_out, n_in]` tensors at load time with no dequantization.
///
/// On Metal the carve is a BORROW (`QTensor::byte_view`): all `n_experts` matrices and the stack
/// itself point into one allocation. That matters because `MoeMode::Fused` keeps the stack alive
/// for `mul_mv_id` while still needing the per-expert matrices for prefill -- copying would put
/// the ~20 GB of 35B expert weights in memory twice. Elsewhere (and if the byte offsets are not
/// suitably aligned) it falls back to copying, which is what this did before.
fn split_stacked_experts(t: &QTensor, device: &Device) -> Result<Vec<QMatMul>> {
    let (n_experts, n_out, n_in) = t.shape().dims3()?;
    let dtype = t.dtype();
    let total = t.storage_size_in_bytes();
    let per_expert = total / n_experts;
    if per_expert * n_experts != total {
        candle::bail!(
            "expert stack {:?} of {total} bytes does not divide evenly into {n_experts} experts",
            t.shape().dims(),
        )
    }

    // Decide the strategy ONCE, from the stride itself -- never by probing expert 0. Offset 0 is
    // trivially aligned for every dtype, so a probe there always succeeds on Metal: it would make
    // the copying fallback below dead code, and a stack whose stride is not alignable would then
    // fail at expert 1 instead of quietly copying. `e * per_expert` is aligned for every e exactly
    // when `per_expert` is, so this one test settles all of them.
    let aliasable = per_expert.is_multiple_of(256) && per_expert.is_multiple_of(dtype.type_size());

    let mut out = Vec::with_capacity(n_experts);
    if aliasable {
        // byte_view still returns None on non-Metal backends, which is the other way into the
        // copying path.
        let mut views = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            match t.byte_view(e * per_expert, per_expert, (n_out, n_in))? {
                Some(view) => views.push(view),
                None => {
                    views.clear();
                    break;
                }
            }
        }
        if views.len() == n_experts {
            for view in views {
                out.push(QMatMul::from_weights(Arc::new(view))?);
            }
            return Ok(out);
        }
    }

    let data = t.data()?;
    for e in 0..n_experts {
        let bytes = &data[e * per_expert..(e + 1) * per_expert];
        let storage = QStorage::from_data(Cow::Borrowed(bytes), device, dtype)?;
        let qt = QTensor::new(storage, (n_out, n_in))?;
        out.push(QMatMul::from_weights(Arc::new(qt))?);
    }
    Ok(out)
}

/// Sparse MoE feed-forward with a gated shared expert.
///
/// Mirrors llama.cpp's `llama_model_qwen35moe::graph::build_layer_ffn`
/// (`src/models/qwen35moe.cpp:497-545`) and `llm_graph_context::build_moe_ffn`
/// (`src/llama-graph.cpp:1914+`), which is the reference the token-id gate compares against:
///
///   probs   = softmax(ffn_gate_inp @ x)          <- softmax BEFORE top-k
///   w       = probs[top_k] / sum(probs[top_k])   <- norm_w = true, sum clamped to >= 6.1035e-5
///   moe     = sum_j w_j * down_e(silu(gate_e(x)) * up_e(x))
///   shexp   = down_shexp(silu(gate_shexp(x)) * up_shexp(x)) * sigmoid(ffn_gate_inp_shexp @ x)
///   out     = moe + shexp
///
/// `expert_weights_scale` is deliberately absent: it defaults to 0.0 in llama.cpp's hparams,
/// `qwen35moe.cpp` never reads it from the GGUF, and `build_moe_ffn` skips the scaling when it is
/// 0.0 or 1.0. Note also that `norm_w` is TRUE here -- `quantized_qwen3_moe.rs` *infers*
/// `norm_topk_prob` from the presence of a shared expert, which would yield `false` for this
/// checkpoint and be wrong.
#[derive(Debug, Clone)]
pub struct MoeWeights {
    /// `[n_experts, hidden]`, f32. Kept dense: it is F32 in the GGUF and tiny.
    router: Tensor,
    gate_experts: Vec<QMatMul>,
    up_experts: Vec<QMatMul>,
    down_experts: Vec<QMatMul>,
    /// `[1, hidden]`, f32. The GGUF stores this 1-D; it produces one scalar gate per token.
    shared_router: Tensor,
    shared_gate: QMatMul,
    shared_up: QMatMul,
    shared_down: QMatMul,
    num_experts_per_tok: usize,
    /// Quant types actually read off THIS layer's tensors, not assumed from another layer.
    expert_quants: ExpertQuants,
    /// The undivided `[n_experts, n_out, n_in]` stacks, kept if EITHER phase asks for
    /// [`MoeMode::Fused`]: the fused kernels address experts as `base + expert_id * stride`,
    /// which needs one allocation. The per-expert `QMatMul`s above are views into these, so
    /// holding both costs no extra memory.
    stacked: Option<StackedExperts>,
    prefill_mode: MoeMode,
    decode_mode: MoeMode,
}

/// The three routed-expert weight stacks, undivided.
#[derive(Debug, Clone)]
struct StackedExperts {
    gate: Arc<QTensor>,
    up: Arc<QTensor>,
    down: Arc<QTensor>,
}

impl MoeWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        num_experts_per_tok: usize,
        modes: MoeModes,
    ) -> Result<Self> {
        let device = gg.device.clone();
        let router = gg
            .tensor(&format!("{prefix}.ffn_gate_inp.weight"))?
            .dequantize(&device)?
            .to_dtype(DType::F32)?;
        let gate_exps_name = format!("{prefix}.ffn_gate_exps.weight");
        let gate_exps_t = gg.tensor(&gate_exps_name)?;
        let gate_dtype = check_expert_quant(&gate_exps_name, gate_exps_t.dtype())?;
        let gate_experts = split_stacked_experts(&gate_exps_t, &device)?;

        let up_exps_name = format!("{prefix}.ffn_up_exps.weight");
        let up_exps_t = gg.tensor(&up_exps_name)?;
        let up_dtype = check_expert_quant(&up_exps_name, up_exps_t.dtype())?;
        let up_experts = split_stacked_experts(&up_exps_t, &device)?;

        let down_exps_name = format!("{prefix}.ffn_down_exps.weight");
        let down_exps_t = gg.tensor(&down_exps_name)?;
        let down_dtype = check_expert_quant(&down_exps_name, down_exps_t.dtype())?;
        let down_experts = split_stacked_experts(&down_exps_t, &device)?;

        let expert_quants = ExpertQuants {
            gate: gate_dtype,
            up: up_dtype,
            down: down_dtype,
        };

        let stacked = if modes.any_fused() {
            Some(StackedExperts {
                gate: Arc::new(gate_exps_t),
                up: Arc::new(up_exps_t),
                down: Arc::new(down_exps_t),
            })
        } else {
            None
        };

        let shared_router = gg
            .tensor(&format!("{prefix}.ffn_gate_inp_shexp.weight"))?
            .dequantize(&device)?
            .to_dtype(DType::F32)?;
        let hidden = shared_router.elem_count();
        let shared_router = shared_router.reshape((1, hidden))?;

        let (shared_gate, _) = gg.qmatmul(&format!("{prefix}.ffn_gate_shexp.weight"))?;
        let (shared_up, _) = gg.qmatmul(&format!("{prefix}.ffn_up_shexp.weight"))?;
        let (shared_down, _) = gg.qmatmul(&format!("{prefix}.ffn_down_shexp.weight"))?;

        Ok(Self {
            router,
            gate_experts,
            up_experts,
            down_experts,
            shared_router,
            shared_gate,
            shared_up,
            shared_down,
            num_experts_per_tok,
            expert_quants,
            stacked,
            prefill_mode: modes.prefill,
            decode_mode: modes.decode,
        })
    }

    /// The quant types this layer's routed-expert tensors were actually read as -- not layer 0's,
    /// not a checkpoint-wide assumption. See [`ExpertQuants`].
    pub fn expert_quants(&self) -> ExpertQuants {
        self.expert_quants
    }

    fn expert_ffn(&self, e: usize, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_experts[e].forward(xs)?)?;
        let up = self.up_experts[e].forward(xs)?;
        self.down_experts[e].forward(&(gate * up)?)
    }

    /// Router logits -> softmax -> top-k -> renormalise. Shared by both modes so the two arms
    /// cannot drift: everything that decides WHICH experts run lives here exactly once.
    ///
    /// Returns `(ids, weights)`, both `[n_tokens, k]`, ids as u32.
    fn route(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = xs.matmul(&self.router.t()?.contiguous()?)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        let k = self.num_experts_per_tok;
        // `narrow` on an argsort is a view; the gather below needs it contiguous.
        let topk_ids = probs
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, k)?
            .contiguous()?;
        let weights = probs.gather(&topk_ids, D::Minus1)?;
        // Clamp exactly as llama.cpp does (llama-graph.cpp:2061) so a degenerate row cannot divide
        // by zero.
        let denom = weights
            .sum_keepdim(D::Minus1)?
            .clamp(6.103515625e-5, f64::INFINITY)?;
        let weights = weights.broadcast_div(&denom)?;
        Ok((topk_ids, weights))
    }

    /// The routed experts, as `n_experts` grouped plain matmuls. Returns `[n_tokens, hidden]`.
    ///
    /// Routing is read back to host because it drives control flow, which forces a GPU sync.
    /// Grouping is what makes this tolerable at all: without it a 5304-token prefill would issue
    /// 42k matmuls per projection per layer.
    fn routed_loop(
        &self,
        xs: &Tensor,
        topk_ids: &Tensor,
        weights: &Tensor,
        n_tokens: usize,
        hidden: usize,
    ) -> Result<Tensor> {
        let k = self.num_experts_per_tok;
        let ids_host = topk_ids.flatten_all()?.to_vec1::<u32>()?;
        let w_host = weights.flatten_all()?.to_vec1::<f32>()?;
        let n_experts = self.gate_experts.len();
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); n_experts];
        let mut coefs: Vec<Vec<f32>> = vec![Vec::new(); n_experts];
        for (slot, (&e, &w)) in ids_host.iter().zip(w_host.iter()).enumerate() {
            let token = (slot / k) as u32;
            rows[e as usize].push(token);
            coefs[e as usize].push(w);
        }

        let device = xs.device();
        let mut out = Tensor::zeros((n_tokens, hidden), DType::F32, device)?;
        for e in 0..n_experts {
            if rows[e].is_empty() {
                continue;
            }
            let idx = Tensor::from_slice(&rows[e], rows[e].len(), device)?;
            let xe = xs.index_select(&idx, 0)?;
            let ye = self.expert_ffn(e, &xe)?;
            let w = Tensor::from_slice(&coefs[e], (rows[e].len(), 1), device)?;
            out = out.index_add(&idx, &ye.broadcast_mul(&w)?, 0)?;
        }
        Ok(out)
    }

    /// The routed experts, as three fused Metal-kernel dispatches (`mul_mv_id` for one token,
    /// `mul_mm_id` for many). Returns `[n_tokens, hidden]`.
    ///
    /// Entirely on device: the routing is never read back, so this loses the sync that
    /// [`MoeWeights::routed_loop`]'s host-side grouping needs. Which underlying kernel runs is
    /// `QMetalStorage::indexed_moe_forward`'s call, not this function's -- it always calls
    /// `indexed_moe_forward` the same way regardless of `n_tokens`.
    ///
    /// The top-k slots are sorted by ASCENDING EXPERT ID before the weighted sum. That is not
    /// cosmetic: `routed_loop` accumulates `out += w_e * y_e` walking `e` from 0 upward, and f32
    /// addition is not associative, so summing the same eight terms in descending-probability
    /// order can differ in the last bit -- enough to flip an argmax and fail the token-id
    /// equivalence gate for a reason that is not a bug in the kernel.
    fn routed_fused(
        &self,
        xs: &Tensor,
        stacked: &StackedExperts,
        topk_ids: &Tensor,
        weights: &Tensor,
        n_tokens: usize,
        hidden: usize,
    ) -> Result<Tensor> {
        let k = self.num_experts_per_tok;
        // arg_sort over the ids themselves; f32 because every expert id (< 2^24) is exact there
        // and candle's arg_sort is defined for it on every backend.
        let order = topk_ids
            .to_dtype(DType::F32)?
            .arg_sort_last_dim(true)?
            .contiguous()?;
        let topk_ids = topk_ids.gather(&order, D::Minus1)?.contiguous()?;
        let weights = weights.gather(&order, D::Minus1)?.contiguous()?;

        // [n_tokens, 1, hidden]: dim 1 of 1 tells the kernel to feed the same token vector to
        // every one of the k slots.
        let x3 = xs.reshape((n_tokens, 1, hidden))?;
        let gate = stacked.gate.indexed_moe_forward(&x3, &topk_ids)?;
        let up = stacked.up.indexed_moe_forward(&x3, &topk_ids)?;
        let h = (candle_nn::ops::silu(&gate)? * up)?.contiguous()?;
        // [n_tokens, k, n_ff]: dim 1 of k tells the kernel each slot has its own input row.
        let y = stacked.down.indexed_moe_forward(&h, &topk_ids)?;
        let w = weights.reshape((n_tokens, k, 1))?;
        y.broadcast_mul(&w)?.sum(1)
    }
}

impl Module for MoeWeights {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, hidden) = x.dims3()?;
        let original_dtype = x.dtype();
        let xs = x.reshape(((), hidden))?.to_dtype(DType::F32)?;
        let n_tokens = b * t;

        let (topk_ids, weights) = self.route(&xs)?;

        // Phase picks its OWN mode: prefill (n_tokens > 1) and decode (n_tokens == 1) are
        // independently configurable (`MoeModes`) precisely because Task 4 measured them not to
        // agree on which mode wins -- `mul_mv_id` decode beats `Loop` decode, but `mul_mm_id`
        // prefill on a many-expert checkpoint LOSES to `Loop` prefill (78.0 vs 118.3 tok/s on the
        // 35B/256-expert, 5304-token case). `QMetalStorage::indexed_moe_forward` still picks
        // `mul_mv_id` vs `mul_mm_id` (and, for `mul_mm_id`, chunks the token axis) internally --
        // this only decides whether `Fused` is asked for at all, for this call's `n_tokens`.
        let phase_mode = if n_tokens == 1 {
            self.decode_mode
        } else {
            self.prefill_mode
        };
        let out = match (phase_mode, self.stacked.as_ref()) {
            (MoeMode::Fused, Some(stacked)) => {
                self.routed_fused(&xs, stacked, &topk_ids, &weights, n_tokens, hidden)?
            }
            _ => self.routed_loop(&xs, &topk_ids, &weights, n_tokens, hidden)?,
        };

        // Shared expert, applied to every token and gated by its own sigmoid scalar.
        let sh_gate = candle_nn::ops::silu(&self.shared_gate.forward(&xs)?)?;
        let sh_up = self.shared_up.forward(&xs)?;
        let sh = self.shared_down.forward(&(sh_gate * sh_up)?)?;
        let sh_scale =
            candle_nn::ops::sigmoid(&xs.matmul(&self.shared_router.t()?.contiguous()?)?)?;
        let out = (out + sh.broadcast_mul(&sh_scale)?)?;

        out.reshape((b, t, hidden))?.to_dtype(original_dtype)
    }
}

/// Dense FFN or sparse MoE, chosen per layer from the GGUF's `expert_count`.
#[derive(Debug, Clone)]
enum Ffn {
    Dense(MlpWeights),
    Moe(MoeWeights),
}

impl Module for Ffn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(m) => m.forward(x),
            Self::Moe(m) => m.forward(x),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    rotary_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(
        // Kept for call-site compatibility but deliberately unused: the angle table is always
        // built in f32. See the comment on the table construction below.
        _dtype: DType,
        head_dim: usize,
        rotary_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
            candle::bail!(
                "invalid Qwen 3.5 rotary dimension {rotary_dim} for head dimension {head_dim}"
            )
        }
        let dim = rotary_dim;
        let max_seq_len = max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        // The angle table MUST be built in f32, never in the model dtype.
        //
        // Storing positions and angles in f16 wrecks RoPE at long context, because the angle
        // `position * inv_freq` grows without bound while f16's absolute resolution grows with
        // magnitude. Measured against an f64 reference, with this model's rope_theta of 1e7:
        //
        //   position |  angle error  |  max cos error
        //          0 |      0        |   0
        //          1 |   0.0002 rad  |   0.0002
        //       2048 |   0.40   rad  |   0.036
        //       5304 |   0.81   rad  |   0.727   <-- on a quantity bounded in [-1, 1]
        //
        // At 5304 the rotation is essentially random, and f16 cannot even distinguish adjacent
        // positions there (spacing is 4), so different tokens collapse onto the same angle. This
        // made candle's Qwen3.5 agree with llama.cpp on only 2 of 128 greedy token ids on the
        // identical GGUF, while a 1-token prompt matched to ~3 decimals -- position 0 has angle 0,
        // which is exact in any precision, so short prompts hid the fault.
        //
        // `apply` still casts to the activation dtype at use. That is fine: sin/cos are bounded by
        // 1, so a late f16 cast costs a bounded ~1e-3, not 0.73.
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
            rotary_dim,
        })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let apply = |x: &Tensor| -> Result<Tensor> {
            let rotated = candle_nn::rotary_emb::rope(
                &x.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?,
                &cos,
                &sin,
            )?;
            if self.rotary_dim == x.dim(D::Minus1)? {
                Ok(rotated)
            } else {
                Tensor::cat(
                    &[
                        &rotated,
                        &x.narrow(
                            D::Minus1,
                            self.rotary_dim,
                            x.dim(D::Minus1)? - self.rotary_dim,
                        )?,
                    ],
                    D::Minus1,
                )
            }
        };
        let q_embed = apply(q)?;
        let k_embed = apply(k)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: ConcatKvCache,
}

impl AttentionWeights {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        _num_heads: usize,
        _num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
    ) -> Result<Self> {
        let (q_proj, q_out) = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let (k_proj, k_out) = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let (v_proj, _) = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let (o_proj, _) = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let num_heads = q_out / (head_dim * 2); // Q + Gate
        let num_kv_heads = k_out / head_dim;
        let num_kv_groups = num_heads / num_kv_heads;

        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

        let kv_cache = ConcatKvCache::new(2);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            kv_cache,
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q_out = self.q_proj.forward(x)?;
        let q_out = q_out.reshape((b, l, self.num_heads, self.head_dim * 2))?;

        let q = q_out.narrow(D::Minus1, 0, self.head_dim)?.transpose(1, 2)?;
        let gate = q_out.narrow(D::Minus1, self.head_dim, self.head_dim)?;

        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;

        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        let (k, v) = self.kv_cache.append(&k, &v)?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            let m_dtype = m.dtype();
            let scores_dtype = scores.dtype();
            let mask = if m_dtype != scores_dtype {
                m.to_dtype(scores_dtype)?
            } else {
                m.clone()
            };
            scores = scores.broadcast_add(&mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        let ctx = ctx.transpose(1, 2)?;
        // Apply sigmoid gate
        let gate_sigmoid =
            candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?.to_dtype(ctx.dtype())?;
        let ctx = ctx.broadcast_mul(&gate_sigmoid)?;

        let reshaped_ctx = ctx.reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&reshaped_ctx)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct GatedDeltaNetWeights {
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    conv_kernel_size: usize,

    in_proj_qkv: QMatMul,
    in_proj_z: QMatMul,
    in_proj_b: QMatMul,
    in_proj_a: QMatMul,
    out_proj: QMatMul,

    conv1d_weight: Tensor,
    dt_bias_f32: Tensor,
    neg_a_f32: Tensor,
    norm_weight: Tensor,
    norm_eps: f64,

    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
}

impl GatedDeltaNetWeights {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        _hidden_size: usize,
        num_v_heads: usize,
        num_k_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_kernel_size: usize,
        rms_norm_eps: f64,
        prefix: &str,
    ) -> Result<Self> {
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;

        let (in_proj_qkv, _) = gg.qmatmul(&format!("{prefix}.attn_qkv.weight"))?;
        let (in_proj_z, _) = gg.qmatmul(&format!("{prefix}.attn_gate.weight"))?;
        let (in_proj_b, _) = gg.qmatmul(&format!("{prefix}.ssm_beta.weight"))?;
        let (in_proj_a, _) = gg.qmatmul(&format!("{prefix}.ssm_alpha.weight"))?;
        let (out_proj, _) = gg.qmatmul(&format!("{prefix}.ssm_out.weight"))?;

        let conv1d_weight = gg
            .tensor(&format!("{prefix}.ssm_conv1d.weight"))?
            .dequantize(&gg.device)?
            .unsqueeze(1)?;

        // NOT a log. The GGUF converter already stores -exp(A_log) in `ssm_a`
        // (llama.cpp conversion/qwen.py:381: `data_torch = -torch.exp(data_torch)`), so this
        // tensor is used VERBATIM as the decay coefficient. llama.cpp does exactly that:
        // src/models/qwen35.cpp:377 `ggml_mul(alpha_softplus, ssm_a)`.
        let neg_a = gg
            .tensor(&format!("{prefix}.ssm_a"))?
            .dequantize(&gg.device)?;
        let dt_bias = gg
            .tensor(&format!("{prefix}.ssm_dt.bias"))?
            .dequantize(&gg.device)?;

        let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?;
        // Applying -exp() here would be the SECOND application: every value in every ssm_a
        // tensor of this checkpoint is already strictly negative (verified across all 18 layers,
        // e.g. blk.0 spans -1.2941..-0.0014, blk.1 reaches -10.5843). Re-mapping x |-> -e^x
        // compresses them all into (-1, 0) AND inverts their order, because the map is
        // monotonically decreasing: the longest-memory head -0.0014 became -0.9986, the
        // strongest decay -10.5843 became -0.000025. That silently swapped every head's memory
        // horizon and cost candle 126 of 128 token ids against llama.cpp on a 5304-token prompt.
        let neg_a_f32 = neg_a.to_dtype(DType::F32)?;

        let norm_weight = gg
            .tensor(&format!("{prefix}.ssm_norm.weight"))?
            .dequantize(&gg.device)?;

        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size,
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            out_proj,
            conv1d_weight,
            dt_bias_f32,
            neg_a_f32,
            norm_weight,
            norm_eps: rms_norm_eps,
            conv_state: None,
            recurrent_state: None,
        })
    }

    fn l2_norm(xs: &Tensor) -> Result<Tensor> {
        let eps = 1e-6;
        let norm = (xs.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?;
        xs.broadcast_div(&norm)
    }

    fn rms_norm_gated(&self, x: &Tensor, g: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let g_f32 = g.to_dtype(DType::F32)?;

        let gate = g_f32.silu()?;
        let norm_x = (x_f32.sqr()?.mean_keepdim(D::Minus1)? + self.norm_eps)?.sqrt()?;
        let x_normed = x_f32.broadcast_div(&norm_x)?;
        (x_normed * gate)?
            .broadcast_mul(&self.norm_weight.to_dtype(DType::F32)?)?
            .to_dtype(x_dtype)
    }

    fn torch_recurrent_gated_delta_rule(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        initial_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let query = Self::l2_norm(query)?;
        let key = Self::l2_norm(key)?;
        let (batch_size, seq_len, num_heads, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(3)?;
        let scale = 1.0 / (query.dim(3)? as f64).sqrt();
        let query = (query * scale)?;

        let mut state = match initial_state {
            Some(state) => state.to_dtype(DType::F32)?,
            None => Tensor::zeros(
                (batch_size, num_heads, k_head_dim, v_head_dim),
                DType::F32,
                query.device(),
            )?,
        };

        let query_f32 = query.to_dtype(DType::F32)?;
        let key_f32 = key.to_dtype(DType::F32)?;
        let value_f32 = value.to_dtype(DType::F32)?;
        let beta_f32 = beta.to_dtype(DType::F32)?;
        let log_g_f32 = g.to_dtype(DType::F32)?;

        let output = if seq_len == 1 {
            let q_t = query_f32.squeeze(1)?;
            let k_t = key_f32.squeeze(1)?;
            let v_t = value_f32.squeeze(1)?;
            let g_t = log_g_f32.squeeze(1)?.exp()?;
            let beta_t = beta_f32.squeeze(1)?;
            sequential_step(&q_t, &k_t, &v_t, &g_t, &beta_t, &mut state)?.unsqueeze(1)?
        } else {
            gated_delta_rule_chunked(
                &query_f32, &key_f32, &value_f32, &log_g_f32, &beta_f32, &mut state,
            )?
        };

        Ok((output, state))
    }

    fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let initial_dtype = hidden_states.dtype();

        let mixed_qkv = self.in_proj_qkv.forward(hidden_states)?.transpose(1, 2)?;

        let z = self.in_proj_z.forward(hidden_states)?;
        let z = z.reshape((batch_size, seq_len, (), self.head_v_dim))?;

        let b = self.in_proj_b.forward(hidden_states)?;
        let a = self.in_proj_a.forward(hidden_states)?;

        // Continue from carried state whenever there IS carried state -- for ANY seq_len, not
        // only seq_len == 1. The old `&& seq_len == 1` silently DISCARDED both the conv window
        // and the recurrent state on a multi-token forward, so a forward that resumed a partly
        // consumed prompt restarted the recurrence from zero and produced fluent, wrong output.
        // That made `--prefill-chunk N` (N > 1, N < prompt_len) wrong, and it blocks any prompt
        // cache: restoring a snapshot is worthless if the next multi-token forward throws it away.
        //
        // Both branches below generalise without further change:
        //   conv: cat([state (k-1 cols), qkv (seq_len cols)]) is k-1+seq_len wide, so a conv1d
        //         with kernel k emits exactly seq_len columns, and the next state is the LAST
        //         k-1 columns -- narrow(2, seq_len, k-1), which reduces to the old
        //         narrow(2, 1, k-1) when seq_len == 1.
        //   recurrent: gated_delta_rule_chunked already seeds its running state from the
        //         `state` argument (`let mut s = state.clone()`), so a non-zero initial state
        //         propagates correctly through the chunked scan.
        let use_precomputed_states = self.conv_state.is_some();

        let mixed_qkv = if use_precomputed_states {
            let conv_state = self.conv_state.as_mut().unwrap();
            let conv_state_data = Tensor::cat(&[conv_state, &mixed_qkv], 2)?;
            // .contiguous() is not cosmetic: narrow shares the storage Arc, so without it the
            // layer would hold the whole [b, conv_dim, seq_len + k-1] f32 buffer alive for its
            // lifetime just to keep k-1 columns. At a 5184-token prefill that is ~170 MB per
            // layer over 30 layers.
            *conv_state = conv_state_data
                .narrow(2, seq_len, self.conv_kernel_size - 1)?
                .contiguous()?;
            let out = conv_state_data.conv1d(&self.conv1d_weight, 0, 1, 1, self.conv_dim)?;
            candle_nn::ops::silu(&out)?
        } else {
            let pad = self.conv_kernel_size - 1;
            let padding = Tensor::zeros(
                (batch_size, self.conv_dim, pad),
                mixed_qkv.dtype(),
                mixed_qkv.device(),
            )?;
            let padded_qkv = Tensor::cat(&[&padding, &mixed_qkv], 2)?;
            // .contiguous() for the same reason as the other branch: keep k-1 columns, not the
            // whole padded prefill buffer they were narrowed out of.
            self.conv_state = Some(padded_qkv.narrow(2, seq_len, pad)?.contiguous()?);
            let out = padded_qkv.conv1d(&self.conv1d_weight, 0, 1, 1, self.conv_dim)?;
            candle_nn::ops::silu(&out)?
        };

        let mixed_qkv = mixed_qkv.transpose(1, 2)?;

        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        let q = q.reshape((batch_size, seq_len, (), self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, (), self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, (), self.head_v_dim))?;

        let beta = candle_nn::ops::sigmoid(&b)?;
        let g = {
            let a_f32 = a.to_dtype(DType::F32)?;
            let a_plus_dt = a_f32.broadcast_add(&self.dt_bias_f32)?;
            let softplus = (a_plus_dt.exp()? + 1.0)?.log()?;
            self.neg_a_f32.broadcast_mul(&softplus)?
        };

        // TILED, not interleaved. When num_v_heads != num_k_heads, llama.cpp broadcasts q/k with
        // `ggml_repeat_4d(q_conv, head_k_dim, num_v_heads, ...)` (src/models/qwen35.cpp:441-445),
        // and ggml's repeat is cyclic (`dst[i] = src[i % src_ne]`), so v-head h reads k-head
        // `h % num_k_heads`. `repeat_interleave` would give `h / repeat_n` instead, permuting the
        // k-head/v-head pairing against the layout the GGUF converter wrote.
        //
        // Invisible on Qwen3.5-0.8B, where group_count == num_v_heads == 16 so repeat_n == 1 and
        // both forms are the identity. LIVE on Qwen3.6-35B-A3B: group_count 16 vs
        // ssm.inner_size/ssm.state_size = 4096/128 = 32, so repeat_n == 2.
        let repeat_n = self.num_v_heads / self.num_k_heads;
        let q = repeat_tiled(&q, repeat_n, 2)?;
        let k = repeat_tiled(&k, repeat_n, 2)?;

        // Derived from `recurrent_state` itself, NOT from `use_precomputed_states` (which reads
        // `conv_state`). Gating one field on the other means a state where `conv_state` is None but
        // `recurrent_state` is Some silently DISCARDS the recurrent state and overwrites it two
        // lines below -- and turn 1 still looks fine, which is the exact failure class this cache
        // work exists to prevent. Unreachable through `forward` alone, but `set_layer_states` is
        // public and can construct it. `as_ref()` is None-safe, so no current path changes.
        let initial_state = self.recurrent_state.as_ref();

        let (core_attn_out, new_state) =
            self.torch_recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, initial_state)?;
        self.recurrent_state = Some(new_state);

        let core_attn_out = core_attn_out.to_dtype(initial_dtype)?;
        let core_attn_out =
            core_attn_out.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
        let core_attn_out = core_attn_out.reshape(((), self.head_v_dim))?;
        let z_flat = z.reshape(((), self.head_v_dim))?;
        let core_attn_out = self.rms_norm_gated(&core_attn_out, &z_flat)?;
        let core_attn_out =
            core_attn_out.reshape((batch_size, seq_len, self.num_v_heads * self.head_v_dim))?;

        self.out_proj.forward(&core_attn_out)
    }

    fn clear_kv_cache(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }
}

/// Repeat along `dim` in TILED order: `[a,b,c]` -> `[a,b,c,a,b,c]`.
///
/// This is ggml's `repeat` semantics (`dst[i] = src[i % src_ne]`) and is what llama.cpp uses to
/// broadcast q/k up to `num_v_heads`. Contrast [`repeat_interleave`], which produces GROUPED order
/// `[a,a,b,b,c,c]` — the two agree only when `repeats == 1`.
fn repeat_tiled(x: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(x.clone());
    }
    let xs = vec![x.clone(); repeats];
    Tensor::cat(&xs, dim)
}

#[allow(dead_code)]
fn repeat_interleave(img: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(img.clone());
    }
    let mut dims = img.dims().to_vec();
    dims[dim] *= repeats;
    let final_dims = dims.clone();
    let img = img.unsqueeze(dim + 1)?;
    let mut expand_dims = img.dims().to_vec();
    expand_dims[dim + 1] = repeats;
    let expanded = img.expand(expand_dims.as_slice())?;
    expanded.reshape(final_dims.as_slice())
}

#[derive(Debug, Clone)]
enum TokenMixer {
    FullAttention(AttentionWeights),
    LinearAttention(GatedDeltaNetWeights),
}

#[derive(Debug, Clone)]
struct LayerWeights {
    token_mixer: TokenMixer,
    mlp: Ffn,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl LayerWeights {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        layer_idx: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        full_attention_interval: usize,
        // Linear attention specifics
        linear_num_key_heads: usize,
        linear_num_value_heads: usize,
        linear_key_head_dim: usize,
        linear_value_head_dim: usize,
        linear_conv_kernel_dim: usize,
        hidden_size: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_modes: MoeModes,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let is_full_attention = (layer_idx + 1).is_multiple_of(full_attention_interval);

        let token_mixer = if is_full_attention {
            TokenMixer::FullAttention(AttentionWeights::new(
                gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary,
                &prefix,
            )?)
        } else {
            TokenMixer::LinearAttention(GatedDeltaNetWeights::new(
                gg,
                hidden_size,
                linear_num_value_heads,
                linear_num_key_heads,
                linear_key_head_dim,
                linear_value_head_dim,
                linear_conv_kernel_dim,
                rms_norm_eps,
                &prefix,
            )?)
        };

        // Every layer is MoE when the checkpoint declares experts. llama.cpp reaches the same
        // conclusion via `decoder_sparse_step` defaulting to 1 (qwen35moe has no such key), so the
        // condition reduces to `expert_count > 0`. Qwen3.6-35B-A3B: 40 of 40 layers.
        let mlp = if num_experts > 0 {
            Ffn::Moe(MoeWeights::new(
                gg,
                &prefix,
                num_experts_per_tok,
                moe_modes,
            )?)
        } else {
            Ffn::Dense(MlpWeights::new(gg, &prefix)?)
        };
        let input_layernorm = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let post_attention_layernorm = gg.rms_norm(
            &format!("{prefix}.post_attention_norm.weight"),
            rms_norm_eps,
        )?;

        Ok(Self {
            token_mixer,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => attn.forward(&x, mask, offset)?,
            TokenMixer::LinearAttention(attn) => attn.forward(&x)?,
        };
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = x.apply(&self.mlp)?;
        x + residual
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => attn.clear_kv_cache(),
            TokenMixer::LinearAttention(attn) => attn.clear_kv_cache(),
        }
    }
}

/// The whole inference state of one layer — everything that makes a forward depend on the tokens
/// already consumed.
///
/// Qwen3.6 mixes two kinds, and they behave very differently under caching:
/// - [`LayerState::Kv`] grows linearly with context (10 of 40 layers here).
/// - [`LayerState::Delta`] is a FIXED-SIZE recurrent state (30 of 40). It is also not invertible:
///   a delta state cannot be truncated back to a shorter prefix the way a KV cache can be sliced,
///   so a cached prefix has to match exactly — there is no partial rewind.
///
/// The tensors are handed over as-is. [`ModelWeights::layer_states`] returns handles that SHARE
/// storage with the model, so a caller keeping a snapshot across forwards must deep-copy them
/// (`Tensor::copy`) rather than assume candle will never gain an in-place cache kernel.
#[derive(Debug, Clone)]
pub enum LayerState {
    /// Full-attention layer: the concatenated K and V, `None` while the cache is empty.
    Kv {
        k: Option<Tensor>,
        v: Option<Tensor>,
    },
    /// GatedDeltaNet layer: the causal-conv window and the recurrent state.
    Delta {
        conv: Option<Tensor>,
        recurrent: Option<Tensor>,
    },
}

impl LayerState {
    /// Total bytes of the tensors held, so a caller can report a snapshot's real size instead of
    /// predicting it from the config.
    pub fn size_in_bytes(&self) -> usize {
        let t = |o: &Option<Tensor>| {
            o.as_ref()
                .map(|t| t.elem_count() * t.dtype().size_in_bytes())
                .unwrap_or(0)
        };
        match self {
            Self::Kv { k, v } => t(k) + t(v),
            Self::Delta { conv, recurrent } => t(conv) + t(recurrent),
        }
    }

    /// A copy that shares NO storage with `self`.
    pub fn deep_copy(&self) -> Result<Self> {
        let c = |o: &Option<Tensor>| -> Result<Option<Tensor>> {
            match o {
                None => Ok(None),
                Some(t) => Ok(Some(t.copy()?)),
            }
        };
        Ok(match self {
            Self::Kv { k, v } => Self::Kv { k: c(k)?, v: c(v)? },
            Self::Delta { conv, recurrent } => Self::Delta {
                conv: c(conv)?,
                recurrent: c(recurrent)?,
            },
        })
    }
}

/// A GatedDeltaNet state must have both halves or neither.
///
/// A forward can never produce a half-set state, but [`ModelWeights::set_layer_states`] is public
/// and could be handed one. Applying it would mean the recurrence resumes while the conv window
/// restarts from zero (or the reverse) -- wrong output, no error, and the first turn still looks
/// fine, which is the failure class this whole cache path exists to prevent.
fn check_delta_state_complete(
    layer: usize,
    conv: &Option<Tensor>,
    recurrent: &Option<Tensor>,
) -> Result<()> {
    if conv.is_some() != recurrent.is_some() {
        candle::bail!(
            "layer {layer}: Delta state is half-set (conv={}, recurrent={}); both must be Some \
             or both None",
            if conv.is_some() { "Some" } else { "None" },
            if recurrent.is_some() { "Some" } else { "None" }
        );
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_with_moe(ct, reader, device, MoeModes::default())
    }

    /// As [`ModelWeights::from_gguf`], but picking how the routed experts are evaluated,
    /// independently for prefill and decode -- see [`MoeModes`].
    ///
    /// The mode is a LOAD-time choice, not a per-call one, because [`MoeMode::Fused`] has to keep
    /// the undivided expert stacks alive for the fused kernels to index into.
    pub fn from_gguf_with_moe<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        moe_modes: MoeModes,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        // Key prefixes come from the file's own `general.architecture`, not from a hardcoded
        // substitution. It is `qwen35` for the dense Qwen3.5 checkpoints and `qwen35moe` for the
        // MoE ones (Qwen3.6-35B-A3B), and the old `replace("qwen3.", "qwen35.")` could never
        // produce the latter -- the 35B failed to load with
        // `cannot find qwen3.attention.head_count or qwen35.attention.head_count`.
        let arch = match gg.metadata().get("general.architecture") {
            Some(v) => v.to_string()?.clone(),
            None => candle::bail!("cannot find general.architecture in metadata"),
        };
        let md_get = |s: &str| {
            let keyed = s.replace("qwen3.", &format!("{arch}."));
            match gg.metadata().get(s).or_else(|| gg.metadata().get(&keyed)) {
                Some(v) => Ok(v),
                None => candle::bail!("cannot find {s} or {keyed} in metadata"),
            }
        };

        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let key_length = md_get("qwen3.attention.key_length")?.to_u32()? as usize;

        let head_dim = key_length;
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen3.rope.freq_base")?.to_f32()? as f64;
        let rotary_dim = md_get("qwen3.rope.dimension_count")
            .and_then(|value| value.to_u32())
            .map(|value| value as usize)
            .unwrap_or(head_dim);

        let full_attention_interval = md_get("qwen3.full_attention_interval")?.to_u32()? as usize;

        // Linear attention specifics
        let linear_num_key_heads = md_get("qwen3.ssm.group_count")?.to_u32()? as usize;
        let ssm_inner_size = md_get("qwen3.ssm.inner_size")?.to_u32()? as usize;
        let linear_key_head_dim = md_get("qwen3.ssm.state_size")?.to_u32()? as usize;
        let linear_value_head_dim = linear_key_head_dim;
        let linear_num_value_heads = ssm_inner_size / linear_value_head_dim;
        let linear_conv_kernel_dim = md_get("qwen3.ssm.conv_kernel")?.to_u32()? as usize;

        // MoE. Absent on the dense checkpoints, so both are optional and default to 0 -> dense.
        let num_experts = md_get("qwen3.expert_count")
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(0);
        let num_experts_per_tok = md_get("qwen3.expert_used_count")
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(0);

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            rotary_dim,
            max_position_embeddings,
            rope_freq_base,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &mut gg,
                i,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                full_attention_interval,
                linear_num_key_heads,
                linear_num_value_heads,
                linear_key_head_dim,
                linear_value_head_dim,
                linear_conv_kernel_dim,
                hidden_size,
                num_experts,
                num_experts_per_tok,
                moe_modes,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
        })
    }

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| (0..(tgt + offset)).map(move |j| if j <= i + offset { 0. } else { minf }))
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };
        for layer in &mut self.layers {
            h = layer.forward(&h, causal_mask.as_ref(), offset)?;
        }
        let h = self.norm.forward(&h)?;
        let last_hidden = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }

    /// Every layer's inference state, in layer order.
    ///
    /// The tensors SHARE storage with the model — this is a borrow made owned, not a copy. Use
    /// [`LayerState::deep_copy`] on the result to keep a snapshot that survives later forwards.
    pub fn layer_states(&self) -> Vec<LayerState> {
        self.layers
            .iter()
            .map(|layer| match &layer.token_mixer {
                TokenMixer::FullAttention(attn) => LayerState::Kv {
                    k: attn.kv_cache.k().cloned(),
                    v: attn.kv_cache.v().cloned(),
                },
                TokenMixer::LinearAttention(attn) => LayerState::Delta {
                    conv: attn.conv_state.clone(),
                    recurrent: attn.recurrent_state.clone(),
                },
            })
            .collect()
    }

    /// Overwrite every layer's inference state.
    ///
    /// The tensors are installed AS GIVEN. A caller that intends to restore the same states again
    /// later must pass deep copies, or the model and the snapshot would share storage.
    ///
    /// Errors if the states do not match the model layer-for-layer: a Kv state landing on a
    /// GatedDeltaNet layer is a caller bug that would otherwise corrupt generation silently.
    pub fn set_layer_states(&mut self, states: &[LayerState]) -> Result<()> {
        if states.len() != self.layers.len() {
            candle::bail!(
                "state has {} layers, model has {}",
                states.len(),
                self.layers.len()
            );
        }
        for (i, (layer, state)) in self.layers.iter_mut().zip(states).enumerate() {
            match (&mut layer.token_mixer, state) {
                (TokenMixer::FullAttention(attn), LayerState::Kv { k, v }) => {
                    attn.kv_cache.reset();
                    if let (Some(k), Some(v)) = (k, v) {
                        // append() on a just-reset cache stores k/v directly (there is nothing to
                        // concatenate with), so this is a set, not a growth step.
                        attn.kv_cache.append(k, v)?;
                    }
                }
                (TokenMixer::LinearAttention(attn), LayerState::Delta { conv, recurrent }) => {
                    check_delta_state_complete(i, conv, recurrent)?;
                    attn.conv_state = conv.clone();
                    attn.recurrent_state = recurrent.clone();
                }
                _ => candle::bail!("layer {i}: state kind does not match the model's layer kind"),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small GatedDeltaNet layer with random dense weights. Built field-by-field on purpose:
    /// `GatedDeltaNetWeights::new` needs a GGUF, and requiring a 22 GB checkpoint is exactly why
    /// this layer had no unit test and why the state-carry bug survived.
    fn tiny_delta_layer(
        hidden: usize,
        num_v_heads: usize,
        num_k_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_kernel_size: usize,
        device: &Device,
    ) -> GatedDeltaNetWeights {
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        // QMatMul::forward computes x @ w.t(), so a weight is [out, in]. GgmlDType::F32 is a
        // pass-through "quantization", which keeps the arithmetic exact -- this test asserts bit
        // equality, so a lossy dtype would blur the very thing being measured.
        let w = |out: usize, r#in: usize| {
            let t = Tensor::randn(0f32, 0.1f32, (out, r#in), device).unwrap();
            QMatMul::from_weights(Arc::new(QTensor::quantize(&t, GgmlDType::F32).unwrap())).unwrap()
        };
        GatedDeltaNetWeights {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size,
            in_proj_qkv: w(conv_dim, hidden),
            in_proj_z: w(value_dim, hidden),
            in_proj_b: w(num_v_heads, hidden),
            in_proj_a: w(num_v_heads, hidden),
            out_proj: w(hidden, value_dim),
            conv1d_weight: Tensor::randn(0f32, 0.1f32, (conv_dim, 1, conv_kernel_size), device)
                .unwrap(),
            dt_bias_f32: Tensor::randn(0f32, 0.1f32, num_v_heads, device).unwrap(),
            // Strictly negative, as the GGUF converter already stores -exp(A_log).
            neg_a_f32: Tensor::full(-0.5f32, num_v_heads, device).unwrap(),
            norm_weight: Tensor::ones(head_v_dim, DType::F32, device).unwrap(),
            norm_eps: 1e-6,
            conv_state: None,
            recurrent_state: None,
        }
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .cloned()
            .fold(0.0f32, f32::max)
    }

    /// A GatedDeltaNet layer must produce the same output whether the prompt is forwarded in one
    /// pass or in two that meet on a `CHUNK_SIZE` boundary and carry state.
    ///
    /// This is the test whose absence let the real bug live: `use_precomputed_states` used to be
    /// `self.conv_state.is_some() && seq_len == 1`, so the second forward here -- multi-token,
    /// with state carried in -- silently threw the conv window and the recurrent state away and
    /// restarted from zero. No error, fluent wrong output. Restoring that condition must make this
    /// test fail; `qwen3_5_linear_attn_scan::tests::prefill_split_matches_undivided_cpu` covers the
    /// scan arithmetic underneath it but cannot see this condition at all.
    ///
    /// Also covers the second half of the same bug class: after the split forward, the layer's
    /// state must equal the undivided layer's state, or the NEXT turn diverges rather than this
    /// one.
    #[test]
    fn delta_layer_split_prefill_matches_undivided_cpu() {
        let device = Device::Cpu;
        let (hidden, t, split) = (32usize, 192usize, 128usize);
        assert_eq!(
            split % crate::models::qwen3_5_linear_attn_scan::CHUNK_SIZE,
            0
        );

        let mut whole = tiny_delta_layer(hidden, 4, 2, 8, 8, 4, &device);
        let mut split_layer = whole.clone();

        let x = Tensor::randn(0f32, 1.0f32, (1, t, hidden), &device).unwrap();

        let out_whole = whole.forward(&x).unwrap();

        let out_a = split_layer
            .forward(&x.narrow(1, 0, split).unwrap())
            .unwrap();
        let out_b = split_layer
            .forward(&x.narrow(1, split, t - split).unwrap())
            .unwrap();
        let out_split = Tensor::cat(&[&out_a, &out_b], 1).unwrap();

        let out_diff = max_abs_diff(&out_whole, &out_split);
        let rec_diff = max_abs_diff(
            whole.recurrent_state.as_ref().unwrap(),
            split_layer.recurrent_state.as_ref().unwrap(),
        );
        let conv_diff = max_abs_diff(
            whole.conv_state.as_ref().unwrap(),
            split_layer.conv_state.as_ref().unwrap(),
        );
        println!(
            "t={t} split={split}: out_diff={out_diff} recurrent_diff={rec_diff} conv_diff={conv_diff}"
        );

        assert_eq!(
            out_diff, 0.0,
            "chunk-aligned split changed the layer output"
        );
        assert_eq!(
            rec_diff, 0.0,
            "chunk-aligned split left a different recurrent state"
        );
        assert_eq!(
            conv_diff, 0.0,
            "chunk-aligned split left a different conv state"
        );
    }

    /// A half-set Delta state must be refused rather than silently half-applied. This is the
    /// guard `set_layer_states` runs on every Delta layer.
    #[test]
    fn half_set_delta_state_is_rejected() {
        let device = Device::Cpu;
        let t = || Some(Tensor::zeros((1, 4, 8, 8), DType::F32, &device).unwrap());

        check_delta_state_complete(3, &None, &None).expect("both None is a cleared layer");
        check_delta_state_complete(3, &t(), &t()).expect("both Some is a live layer");

        for (conv, rec) in [(t(), None), (None, t())] {
            let err = check_delta_state_complete(3, &conv, &rec)
                .expect_err("half-set Delta state must be refused");
            let msg = err.to_string();
            // The message has to name BOTH fields: "invalid state" would leave the caller
            // guessing which half it forgot.
            assert!(msg.contains("conv=") && msg.contains("recurrent="), "{msg}");
            assert!(msg.contains("layer 3"), "{msg}");
        }
    }

    /// The 35B's `ffn_down_exps` is unsloth's dynamic quant: Q5_K on 37 layers, Q6_K on 3. A
    /// loader that reads the type once (e.g. from layer 0) and reuses it for all 40 layers will
    /// report a single type here instead of two. The 14B rung cannot catch this -- it is plain
    /// Q4_K_M/Q6_K with no per-layer variation -- so this test requires the 35B specifically.
    ///
    /// `QWEN36_35B_GGUF` is read at RUNTIME (`std::env::var`), not baked in via `env!` at compile
    /// time: the test must not force a rebuild every time the checkpoint path changes, and must
    /// not silently pass/skip when the variable or file is missing -- it fails loudly instead, as
    /// a test that can silently skip is a test that can never fail.
    #[test]
    fn expert_quants_are_read_per_layer_not_once() {
        let path = match std::env::var("QWEN36_35B_GGUF") {
            Ok(path) => path,
            Err(_) => panic!("QWEN36_35B_GGUF env var required for this test (35B GGUF path)"),
        };
        let path = std::path::Path::new(&path);
        if !path.exists() {
            panic!("35B GGUF required for this test: {}", path.display());
        }
        let quants = read_expert_quants_per_layer(path)
            .expect("failed to read expert quant types from the 35B GGUF");
        assert_eq!(quants.len(), 40);

        let downs: std::collections::HashSet<_> = quants.iter().map(|q| q.down).collect();
        assert!(
            downs.len() > 1,
            "expected ffn_down_exps to carry MORE THAN ONE quant type across layers, got {downs:?} \
             -- if this fails the loader is collapsing per-layer types"
        );
        assert_eq!(
            quants.iter().filter(|q| q.down == GgmlDType::Q6K).count(),
            3
        );
        assert_eq!(
            quants.iter().filter(|q| q.down == GgmlDType::Q5K).count(),
            37
        );
        assert!(quants
            .iter()
            .all(|q| q.gate == GgmlDType::Q4K && q.up == GgmlDType::Q4K));

        // Assert layer IDENTITY, not just the multiset. A loader that reads the correct set of
        // types but assigns them to the wrong layer indices (an off-by-one in the loop, a
        // shuffle) would still pass every assertion above -- Tasks 3/4/12 dispatch a Metal kernel
        // PER LAYER using exactly this value, so a misattribution bug must be caught here.
        //
        // Ground truth independently verified with llama.cpp's own gguf-py reader (NOT this
        // crate's gguf parser, so this isn't circular):
        //   PYTHONPATH=llama.cpp/gguf-py python3 -c
        //     'import gguf; r = gguf.GGUFReader("...35B.gguf"); ...'
        //   -> Q6_K on blk.{34,38,39}.ffn_down_exps.weight, Q5_K on all other 37 layers.
        let q6_layers: Vec<usize> = quants
            .iter()
            .enumerate()
            .filter(|(_, q)| q.down == GgmlDType::Q6K)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            q6_layers,
            vec![34, 38, 39],
            "Q6_K ffn_down_exps must land on exactly layers 34, 38, 39 -- got {q6_layers:?}; a \
             correct multiset at the wrong layer indices is exactly the misattribution bug this \
             test exists to catch"
        );
    }
}
