//! Gemma 4 dense text tower (`google/gemma-4-31B-it`) from a GGUF, with quantized weights.
//!
//! The maths follows llama.cpp's `src/models/gemma4.cpp` -- the oracle this port is gated
//! against token for token -- and NOT upstream's `models/gemma4/text.rs` where the two differ.
//! The differences, each checked against gemma4.cpp:
//!
//! - attention scale is 1.0 (`f_attention_scale = 1.0f`), not `1/sqrt(head_dim)`;
//! - norm weights are used as stored: the converter's `norm_shift` is 0 for Gemma 4, so there is
//!   no `+ 1` on the weight;
//! - a key is visible to a sliding-window query iff `0 <= q - k < window`
//!   (`LLAMA_SWA_TYPE_STANDARD` masks `p1 - p0 >= n_swa`); text.rs lets `window + 1` through;
//! - `attention_k_eq_v`: a layer without `attn_v` uses the RAW K projection as V (before
//!   `attn_k_norm` and before RoPE), which then gets a weightless RMS norm;
//! - `layer_output_scale` multiplies each layer's output;
//! - full-attention layers rotate all `key_length` dims with per-pair frequency factors from
//!   `rope_freqs.weight` (1e30 on the non-rotated pairs), NEOX layout.
//!
//! Only the dense text tower: a checkpoint whose metadata or tensors enable MoE, per-layer-input
//! embeddings, shared KV layers or a vision/audio tower is REFUSED, never silently half-loaded.

use super::with_tracing::QMatMul;
use crate::quantized_nn::RmsNorm;
use crate::utils::repeat_kv;
use candle::quantized::{gguf_file, QTensor};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::Embedding;
use std::collections::HashMap;
use std::io::{Read, Seek};
use std::sync::Arc;

pub const ARCH: &str = "gemma4";

/// RoPE tables are built for `min(context_length, MAX_POSITIONS)` positions. The correction
/// prompt is ~6-8k tokens; 262 144 positions (the checkpoint's context) would cost ~0.8 GB of
/// sin/cos tables for nothing. A longer prompt is an ERROR (`Rope::apply`), never a wrap.
pub const MAX_POSITIONS: usize = 32_768;

type Metadata = HashMap<String, gguf_file::Value>;

fn key(k: &str) -> String {
    format!("{ARCH}.{k}")
}

fn md<'a>(m: &'a Metadata, k: &str) -> Result<&'a gguf_file::Value> {
    let full = key(k);
    match m.get(&full) {
        Some(v) => Ok(v),
        None => candle::bail!("gemma4 GGUF lacks metadata key {full}"),
    }
}

/// Reads any GGUF integer type (U8/I8/U16/I16/U32/I32/U64/I64) as a non-negative `u64`.
///
/// The real Gemma 4 GGUF stores `attention.head_count_kv` as an ARRAY of INT32 (gguf-py's
/// default for int lists), while scalar integers (`block_count`, `sliding_window`, ...) are
/// UINT32. `Value::to_u32()` only matches `Value::U32` exactly (no upcast), so it would refuse
/// the I32 array elements in the real file. Every integer metadata read -- scalar or array
/// element -- goes through this ONE helper instead, so both widths parse.
fn gguf_uint(v: &gguf_file::Value) -> Result<u64> {
    use gguf_file::Value as V;
    match v {
        V::U8(x) => Ok(*x as u64),
        V::U16(x) => Ok(*x as u64),
        V::U32(x) => Ok(*x as u64),
        V::U64(x) => Ok(*x),
        V::I8(x) if *x >= 0 => Ok(*x as u64),
        V::I16(x) if *x >= 0 => Ok(*x as u64),
        V::I32(x) if *x >= 0 => Ok(*x as u64),
        V::I64(x) if *x >= 0 => Ok(*x as u64),
        V::I8(_) | V::I16(_) | V::I32(_) | V::I64(_) => {
            candle::bail!("expected a non-negative integer metadata value, got {v:?}")
        }
        v => candle::bail!("expected an integer metadata value, got {v:?}"),
    }
}

fn md_usize(m: &Metadata, k: &str) -> Result<usize> {
    Ok(gguf_uint(md(m, k)?)? as usize)
}

fn md_f64(m: &Metadata, k: &str) -> Result<f64> {
    Ok(md(m, k)?.to_f32()? as f64)
}

/// Absent counts as 0: the dense checkpoints do not write MoE / PLE keys at all.
fn md_usize_or_zero(m: &Metadata, k: &str) -> Result<usize> {
    match m.get(&key(k)) {
        None => Ok(0),
        Some(v) => Ok(gguf_uint(v)? as usize),
    }
}

/// A per-layer value that llama.cpp accepts either as a scalar or as an `n`-entry array.
fn md_per_layer_usize(m: &Metadata, k: &str, n: usize) -> Result<Vec<usize>> {
    match md(m, k)? {
        gguf_file::Value::Array(vs) => {
            if vs.len() != n {
                candle::bail!("{} has {} entries for {n} layers", key(k), vs.len())
            }
            vs.iter().map(|v| Ok(gguf_uint(v)? as usize)).collect()
        }
        v => Ok(vec![gguf_uint(v)? as usize; n]),
    }
}

fn md_per_layer_bool(m: &Metadata, k: &str, n: usize) -> Result<Vec<bool>> {
    match md(m, k)? {
        gguf_file::Value::Array(vs) if vs.len() == n => vs.iter().map(|v| v.to_bool()).collect(),
        _ => candle::bail!("{} must be a {n}-entry bool array", key(k)),
    }
}

/// Everything the text tower needs from the metadata, per layer where llama.cpp allows it.
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4Config {
    pub block_count: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub head_count: Vec<usize>,
    pub head_count_kv: Vec<usize>,
    /// Head dim of the full-attention layers (`attention.key_length`).
    pub key_length: usize,
    /// Head dim of the sliding layers (`attention.key_length_swa`).
    pub key_length_swa: usize,
    pub is_swa: Vec<bool>,
    pub sliding_window: usize,
    pub rms_norm_eps: f64,
    pub rope_freq_base: f64,
    pub rope_freq_base_swa: f64,
    pub rope_dimension_count: usize,
    pub rope_dimension_count_swa: usize,
    pub final_logit_softcapping: Option<f64>,
    pub context_length: usize,
}

impl Gemma4Config {
    pub fn from_metadata(m: &Metadata) -> Result<Self> {
        let n = md_usize(m, "block_count")?;
        let key_length = md_usize(m, "attention.key_length")?;
        let key_length_swa = md_usize(m, "attention.key_length_swa")?;
        // gemma4.cpp throws on either inequality; so do we.
        if md_usize(m, "attention.value_length")? != key_length
            || md_usize(m, "attention.value_length_swa")? != key_length_swa
        {
            candle::bail!("gemma4 requires key_length == value_length on both layer kinds")
        }
        let rope_dimension_count = md_usize(m, "rope.dimension_count")?;
        let rope_dimension_count_swa = md_usize(m, "rope.dimension_count_swa")?;
        if rope_dimension_count != key_length || rope_dimension_count_swa != key_length_swa {
            candle::bail!(
                "partial rotary dims are not supported: rope.dimension_count {rope_dimension_count} \
                 vs key_length {key_length}, rope.dimension_count_swa {rope_dimension_count_swa} \
                 vs key_length_swa {key_length_swa}"
            )
        }
        if matches!(m.get(&key("feed_forward_length")), Some(gguf_file::Value::Array(_))) {
            candle::bail!("a per-layer feed_forward_length (use_double_wide_mlp) is not supported")
        }
        let sliding_window = md_usize(m, "attention.sliding_window")?;
        if sliding_window == 0 {
            candle::bail!("{} is 0; the sliding layers need a window of at least 1", key("attention.sliding_window"))
        }
        let final_logit_softcapping = match m.get(&key("final_logit_softcapping")) {
            None => None,
            Some(v) => Some(v.to_f32()? as f64),
        };
        Ok(Self {
            block_count: n,
            embedding_length: md_usize(m, "embedding_length")?,
            feed_forward_length: md_usize(m, "feed_forward_length")?,
            head_count: md_per_layer_usize(m, "attention.head_count", n)?,
            head_count_kv: md_per_layer_usize(m, "attention.head_count_kv", n)?,
            key_length,
            key_length_swa,
            is_swa: md_per_layer_bool(m, "attention.sliding_window_pattern", n)?,
            sliding_window,
            rms_norm_eps: md_f64(m, "attention.layer_norm_rms_epsilon")?,
            rope_freq_base: md_f64(m, "rope.freq_base")?,
            // REQUIRED, not defaulted: llama.cpp falls back to rope.freq_base when this key is
            // missing, which would rotate the sliding layers 100x too slowly.
            rope_freq_base_swa: md_f64(m, "rope.freq_base_swa")?,
            rope_dimension_count,
            rope_dimension_count_swa,
            final_logit_softcapping,
            context_length: md_usize(m, "context_length")?,
        })
    }

    pub fn head_dim(&self, layer: usize) -> usize {
        if self.is_swa[layer] {
            self.key_length_swa
        } else {
            self.key_length
        }
    }
}

/// Refuse anything this loader would otherwise half-load in silence.
pub fn check_supported(ct: &gguf_file::Content) -> Result<()> {
    let arch = match ct.metadata.get("general.architecture") {
        Some(v) => v.to_string()?.clone(),
        None => candle::bail!("cannot find general.architecture in metadata"),
    };
    if arch != ARCH {
        candle::bail!("not a {ARCH} GGUF: general.architecture is {arch:?}")
    }
    if let Some(v) = ct.metadata.get("split.count") {
        let count = gguf_uint(v)?;
        if count > 1 {
            candle::bail!(
                "this is one shard of a split GGUF (split.count = {count}); candle reads a single \
                 file -- merge it first with llama-gguf-split --merge"
            )
        }
    }
    let names: Vec<&String> = ct.tensor_infos.keys().collect();
    let experts = md_usize_or_zero(&ct.metadata, "expert_count")?;
    if experts > 0 || names.iter().any(|n| n.contains(".ffn_gate_inp.") || n.contains("_exps.")) {
        candle::bail!(
            "MoE Gemma 4 (expert_count = {experts}) is not supported: this loader has no expert \
             branch and would silently skip the expert weights"
        )
    }
    let ple = md_usize_or_zero(&ct.metadata, "embedding_length_per_layer_input")?;
    if ple > 0 || names.iter().any(|n| n.starts_with("per_layer_")) {
        candle::bail!("per-layer-input embeddings (E2B/E4B, width {ple}) are not supported")
    }
    let shared = md_usize_or_zero(&ct.metadata, "attention.shared_kv_layers")?;
    if shared > 0 {
        candle::bail!("shared KV layers ({shared}) are not supported")
    }
    if let Some(n) = names
        .iter()
        .find(|n| n.starts_with("v.") || n.starts_with("a.") || n.starts_with("mm."))
    {
        candle::bail!("vision/audio tower tensors are present ({n}); only the text tower is supported")
    }
    Ok(())
}

/// A full ring (`window` slots, position `p` in slot `p % window`) in chronological order. A
/// buffer that has not filled yet is already chronological and comes back as is.
fn ring_to_chronological(buf: &Tensor, write_pos: usize, window: usize) -> Result<Tensor> {
    let n = buf.dim(2)?;
    if n < window {
        if write_pos != n {
            candle::bail!(
                "a sliding buffer holding {n} of {window} slots must be written at slot {n}, not {write_pos}"
            )
        }
        return Ok(buf.clone());
    }
    if n != window {
        candle::bail!("a sliding buffer holds {n} slots, more than its window {window}")
    }
    if write_pos == 0 {
        return Ok(buf.clone());
    }
    Tensor::cat(&[&buf.narrow(2, write_pos, window - write_pos)?, &buf.narrow(2, 0, write_pos)?], 2)
}

/// Keep the last `window` of `chrono` (whose last key is position `end_offset - 1`) as a ring.
/// Returns the ring and its write position (`end_offset % window`).
fn chronological_to_ring(chrono: &Tensor, end_offset: usize, window: usize) -> Result<(Tensor, usize)> {
    let total = chrono.dim(2)?;
    if total > end_offset {
        candle::bail!("{total} keys cannot end at position {end_offset}")
    }
    let n = total.min(window);
    let tail = chrono.narrow(2, total - n, n)?;
    if n < window {
        if end_offset != n {
            candle::bail!("{n} keys kept after {end_offset} positions: a filled window cannot shrink")
        }
        return Ok((tail.contiguous()?, end_offset % window));
    }
    // tail[i] is position end_offset - window + i, which belongs in slot (s0 + i) % window.
    let s0 = (end_offset - window) % window;
    let ring = if s0 == 0 {
        tail.contiguous()?
    } else {
        Tensor::cat(&[&tail.narrow(2, window - s0, s0)?, &tail.narrow(2, 0, window - s0)?], 2)?
    };
    Ok((ring, end_offset % window))
}

/// `l x n` additive mask: queries at positions `q_start..q_start+l`, keys at `k_start..k_start+n`.
/// Visible iff the key is not in the future and, with a window, `q - k < window`
/// (llama.cpp `LLAMA_SWA_TYPE_STANDARD`: `p1 - p0 >= n_swa` is masked).
fn mask_values(q_start: usize, l: usize, k_start: usize, n: usize, window: Option<usize>) -> Vec<f32> {
    let mut m = Vec::with_capacity(l * n);
    for i in 0..l {
        let q = q_start + i;
        for j in 0..n {
            let k = k_start + j;
            let visible = k <= q && window.is_none_or(|w| q - k < w);
            m.push(if visible { 0.0 } else { f32::NEG_INFINITY });
        }
    }
    m
}

/// A sliding state must say where it is: `slots == min(offset, window)` and
/// `write_pos == offset % window`. Restoring the tensors of a wrapped ring without its position
/// is exactly what silently broke the Python rig on 2026-09-25; here it is an error.
fn check_sliding_state(
    layer: usize,
    k: &Option<Tensor>,
    v: &Option<Tensor>,
    write_pos: usize,
    offset: usize,
    window: usize,
) -> Result<()> {
    let slots = match (k, v) {
        (None, None) => 0,
        (Some(k), Some(v)) => {
            if k.dims() != v.dims() {
                candle::bail!("layer {layer}: sliding K {:?} and V {:?} differ in shape", k.dims(), v.dims())
            }
            k.dim(2)?
        }
        _ => candle::bail!("layer {layer}: sliding state is half-set; K and V must both be Some or both None"),
    };
    if slots != offset.min(window) || write_pos != offset % window {
        candle::bail!(
            "layer {layer}: sliding state is inconsistent: {slots} slots, write_pos {write_pos}, \
             offset {offset}, window {window} (expected {} slots, write_pos {})",
            offset.min(window),
            offset % window
        )
    }
    Ok(())
}

/// One layer's inference state, public so `s2t-llm` can snapshot and restore a prompt prefix.
///
/// `Sliding` carries its ring's write position and the absolute number of positions seen, BY
/// CONSTRUCTION -- the tensors alone do not say which slot is oldest once the ring has wrapped.
/// The tensors returned by [`ModelWeights::layer_states`] SHARE storage with the model, and the
/// sliding ring is overwritten IN PLACE on decode: keep a snapshot only via [`LayerState::deep_copy`].
#[derive(Debug, Clone)]
pub enum LayerState {
    Sliding { k: Option<Tensor>, v: Option<Tensor>, write_pos: usize, offset: usize },
    Full { k: Option<Tensor>, v: Option<Tensor> },
}

impl LayerState {
    pub fn size_in_bytes(&self) -> usize {
        let t = |o: &Option<Tensor>| o.as_ref().map(|t| t.elem_count() * t.dtype().size_in_bytes()).unwrap_or(0);
        match self {
            Self::Sliding { k, v, .. } | Self::Full { k, v } => t(k) + t(v),
        }
    }

    pub fn deep_copy(&self) -> Result<Self> {
        let c = |o: &Option<Tensor>| -> Result<Option<Tensor>> { o.as_ref().map(|t| t.copy()).transpose() };
        Ok(match self {
            Self::Sliding { k, v, write_pos, offset } => {
                Self::Sliding { k: c(k)?, v: c(v)?, write_pos: *write_pos, offset: *offset }
            }
            Self::Full { k, v } => Self::Full { k: c(k)?, v: c(v)? },
        })
    }

    /// Positions this layer has consumed.
    pub fn position(&self) -> Result<usize> {
        match self {
            Self::Sliding { offset, .. } => Ok(*offset),
            Self::Full { k, .. } => Ok(k.as_ref().map(|t| t.dim(2)).transpose()?.unwrap_or(0)),
        }
    }
}

/// RoPE sin/cos tables, built in f32 (the Qwen3.5 port measured what an f16 angle table does at
/// position 5304: a random rotation). NEOX layout via `candle_nn::rotary_emb::rope`, as llama.cpp
/// uses for gemma4 (`LLAMA_ROPE_TYPE_NEOX`).
#[derive(Debug, Clone)]
struct Rope {
    sin: Tensor,
    cos: Tensor,
}

impl Rope {
    /// `inv_freq[i] = 1 / (base^(2i/n_rot) * freq_factor[i])` -- ggml's `theta / freq_factor`.
    fn new(n_rot: usize, base: f64, freq_factors: Option<&[f32]>, max_pos: usize, dev: &Device) -> Result<Self> {
        let half = n_rot / 2;
        if let Some(ff) = freq_factors {
            if ff.len() != half {
                candle::bail!("rope_freqs has {} entries, a {n_rot}-dim rotary needs {half}", ff.len())
            }
        }
        let inv: Vec<f32> = (0..half)
            .map(|i| {
                let ff = freq_factors.map_or(1.0, |f| f[i] as f64);
                (1.0 / (base.powf(2.0 * i as f64 / n_rot as f64) * ff)) as f32
            })
            .collect();
        let inv = Tensor::from_vec(inv, (1, half), dev)?;
        let t = Tensor::arange(0u32, max_pos as u32, dev)?.to_dtype(DType::F32)?.reshape((max_pos, 1))?;
        let freqs = t.matmul(&inv)?;
        Ok(Self { sin: freqs.sin()?, cos: freqs.cos()? })
    }

    /// `x` is `[b, heads, l, head_dim]`.
    fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let l = x.dim(2)?;
        let max = self.cos.dim(0)?;
        if offset + l > max {
            candle::bail!("position {} is beyond the {max} positions the RoPE tables hold (MAX_POSITIONS)", offset + l)
        }
        let cos = self.cos.narrow(0, offset, l)?.to_dtype(x.dtype())?;
        let sin = self.sin.narrow(0, offset, l)?.to_dtype(x.dtype())?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }
}

/// RMS norm without a learned weight (gemma4.cpp: `ggml_rms_norm(Vcur, eps)`).
fn rms_norm_plain(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
    x.broadcast_div(&(ms + eps)?.sqrt()?)?.to_dtype(dt)
}

/// Scale 1.0: Gemma 4 relies on q_norm/k_norm (gemma4.cpp `f_attention_scale = 1.0f`).
fn attend(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>, n_rep: usize) -> Result<Tensor> {
    let k = repeat_kv(k.clone(), n_rep)?.contiguous()?;
    let v = repeat_kv(v.clone(), n_rep)?.contiguous()?;
    let mut scores = q.contiguous()?.matmul(&k.t()?)?;
    if let Some(m) = mask {
        scores = scores.broadcast_add(m)?;
    }
    candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)
}

struct Masks {
    full: Option<Tensor>,
    sliding: Option<Tensor>,
}

#[derive(Debug, Clone)]
enum KvState {
    Sliding { k: Option<Tensor>, v: Option<Tensor>, write_pos: usize, offset: usize },
    Full(ConcatKvCache),
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    /// `None` = attention_k_eq_v: V is the raw K projection.
    v_proj: Option<QMatMul>,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f64,
    window: usize,
    rope: Arc<Rope>,
    kv: KvState,
}

impl Attention {
    fn forward(&mut self, x: &Tensor, offset: usize, masks: &Masks) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv_heads);
        let q = self.q_proj.forward(x)?.reshape((b, l, nh, hd))?;
        let q = self.rope.apply(&self.q_norm.forward(&q)?.transpose(1, 2)?, offset)?;
        let k_raw = self.k_proj.forward(x)?;
        // k_eq_v: gemma4.cpp assigns `Vcur = Kcur` BEFORE k_norm and RoPE touch Kcur.
        let v_raw = match &self.v_proj {
            Some(v) => v.forward(x)?,
            None => k_raw.clone(),
        };
        let k = k_raw.reshape((b, l, nkv, hd))?;
        let k = self.rope.apply(&self.k_norm.forward(&k)?.transpose(1, 2)?, offset)?;
        let v = rms_norm_plain(&v_raw.reshape((b, l, nkv, hd))?, self.eps)?.transpose(1, 2)?.contiguous()?;
        let (keys, values, mask) = self.store(k, v, offset, l, masks)?;
        let ctx = attend(&q, &keys, &values, mask.as_ref(), nh / nkv)?;
        self.o_proj.forward(&ctx.transpose(1, 2)?.reshape((b, l, nh * hd))?)
    }

    /// Store this forward's K/V; return what the queries attend over and the mask for it.
    fn store(&mut self, k: Tensor, v: Tensor, offset: usize, l: usize, masks: &Masks) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let window = self.window;
        match &mut self.kv {
            KvState::Full(cache) => {
                let (k, v) = cache.append(&k, &v)?;
                Ok((k, v, masks.full.clone()))
            }
            KvState::Sliding { k: sk, v: sv, write_pos, offset: seen } => {
                if *seen != offset {
                    candle::bail!("sliding layer has seen {seen} positions but this forward starts at {offset}")
                }
                let ring_is_full = match sk.as_ref() {
                    Some(t) => t.dim(2)? == window,
                    None => false,
                };
                if l == 1 && ring_is_full {
                    // Decode: overwrite the oldest slot IN PLACE. Every key left is within the
                    // window of this query, so no mask; attention is order-independent.
                    let (Some(rk), Some(rv)) = (sk.as_ref(), sv.as_ref()) else {
                        candle::bail!("sliding state is half-set")
                    };
                    rk.slice_set(&k, 2, *write_pos)?;
                    rv.slice_set(&v, 2, *write_pos)?;
                    *write_pos = (*write_pos + 1) % window;
                    *seen += 1;
                    return Ok((rk.clone(), rv.clone(), None));
                }
                let (kc, vc) = match (sk.take(), sv.take()) {
                    (Some(rk), Some(rv)) => (
                        Tensor::cat(&[&ring_to_chronological(&rk, *write_pos, window)?, &k], 2)?,
                        Tensor::cat(&[&ring_to_chronological(&rv, *write_pos, window)?, &v], 2)?,
                    ),
                    (None, None) => (k, v),
                    _ => candle::bail!("sliding state is half-set"),
                };
                if kc.dim(2)? != offset.min(window) + l {
                    candle::bail!("sliding keys {} != min(offset {offset}, window {window}) + {l}", kc.dim(2)?)
                }
                let end = offset + l;
                let (rk, wp) = chronological_to_ring(&kc, end, window)?;
                let (rv, _) = chronological_to_ring(&vc, end, window)?;
                *sk = Some(rk);
                *sv = Some(rv);
                *write_pos = wp;
                *seen = end;
                Ok((kc, vc, masks.sliding.clone()))
            }
        }
    }

    fn clear(&mut self) {
        match &mut self.kv {
            KvState::Full(c) => c.reset(),
            KvState::Sliding { k, v, write_pos, offset } => {
                (*k, *v, *write_pos, *offset) = (None, None, 0, 0);
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl Module for Mlp {
    /// gelu (tanh approximation, = ggml_gelu / `gelu_pytorch_tanh`) gated, LLM_FFN_PAR.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(x)?.gelu()?;
        self.down.forward(&(g * self.up.forward(x)?)?)
    }
}

#[derive(Debug, Clone)]
struct Layer {
    attn_norm: RmsNorm,
    attn: Attention,
    post_attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    mlp: Mlp,
    post_ffw_norm: RmsNorm,
    out_scale: Option<Tensor>,
}

impl Layer {
    fn forward(&mut self, x: &Tensor, offset: usize, masks: &Masks) -> Result<Tensor> {
        let a = self.attn.forward(&self.attn_norm.forward(x)?, offset, masks)?;
        let h = (self.post_attn_norm.forward(&a)? + x)?;
        let f = self.mlp.forward(&self.ffn_norm.forward(&h)?)?;
        let out = (self.post_ffw_norm.forward(&f)? + &h)?;
        match &self.out_scale {
            Some(s) => out.broadcast_mul(s),
            None => Ok(out),
        }
    }
}

struct Loader<'a, R: Read + Seek> {
    ct: &'a gguf_file::Content,
    reader: &'a mut R,
    device: &'a Device,
}

impl<R: Read + Seek> Loader<'_, R> {
    fn has(&self, name: &str) -> bool {
        self.ct.tensor_infos.contains_key(name)
    }

    fn qtensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(self.reader, name, self.device)
    }

    fn qmatmul(&mut self, name: &str, out: usize, inp: usize) -> Result<QMatMul> {
        let t = self.qtensor(name)?;
        if t.shape().dims() != [out, inp] {
            candle::bail!("{name} has shape {:?}, expected [{out}, {inp}]", t.shape().dims())
        }
        QMatMul::from_weights(Arc::new(t))
    }

    fn norm(&mut self, name: &str, dim: usize, eps: f64) -> Result<RmsNorm> {
        let t = self.qtensor(name)?;
        if t.shape().dims() != [dim] {
            candle::bail!("{name} has shape {:?}, expected [{dim}]", t.shape().dims())
        }
        RmsNorm::from_qtensor(t, eps)
    }

    fn layer(&mut self, cfg: &Gemma4Config, i: usize, rope: Arc<Rope>) -> Result<Layer> {
        let (h, ff, eps) = (cfg.embedding_length, cfg.feed_forward_length, cfg.rms_norm_eps);
        let (hd, nh, nkv) = (cfg.head_dim(i), cfg.head_count[i], cfg.head_count_kv[i]);
        if nkv == 0 || nh % nkv != 0 {
            candle::bail!("layer {i}: {nh} query heads cannot share {nkv} KV heads")
        }
        let n = |s: &str| format!("blk.{i}.{s}.weight");
        let v_proj = if self.has(&n("attn_v")) { Some(self.qmatmul(&n("attn_v"), nkv * hd, h)?) } else { None };
        let kv = if cfg.is_swa[i] {
            KvState::Sliding { k: None, v: None, write_pos: 0, offset: 0 }
        } else {
            KvState::Full(ConcatKvCache::new(2))
        };
        let attn = Attention {
            q_proj: self.qmatmul(&n("attn_q"), nh * hd, h)?,
            k_proj: self.qmatmul(&n("attn_k"), nkv * hd, h)?,
            v_proj,
            o_proj: self.qmatmul(&n("attn_output"), h, nh * hd)?,
            q_norm: self.norm(&n("attn_q_norm"), hd, eps)?,
            k_norm: self.norm(&n("attn_k_norm"), hd, eps)?,
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
            eps,
            window: cfg.sliding_window,
            rope,
            kv,
        };
        let out_scale = if self.has(&n("layer_output_scale")) {
            Some(self.qtensor(&n("layer_output_scale"))?.dequantize(self.device)?)
        } else {
            None
        };
        Ok(Layer {
            attn_norm: self.norm(&n("attn_norm"), h, eps)?,
            attn,
            post_attn_norm: self.norm(&n("post_attention_norm"), h, eps)?,
            ffn_norm: self.norm(&n("ffn_norm"), h, eps)?,
            mlp: Mlp {
                gate: self.qmatmul(&n("ffn_gate"), ff, h)?,
                up: self.qmatmul(&n("ffn_up"), ff, h)?,
                down: self.qmatmul(&n("ffn_down"), h, ff)?,
            },
            post_ffw_norm: self.norm(&n("post_ffw_norm"), h, eps)?,
            out_scale,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed: Embedding,
    embed_scale: f64,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: QMatMul,
    softcap: Option<f64>,
    pos: usize,
    device: Device,
    cfg: Gemma4Config,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(ct: gguf_file::Content, reader: &mut R, device: &Device) -> Result<Self> {
        check_supported(&ct)?;
        let cfg = Gemma4Config::from_metadata(&ct.metadata)?;
        let mut ld = Loader { ct: &ct, reader, device };
        let (h, eps) = (cfg.embedding_length, cfg.rms_norm_eps);
        let max_pos = cfg.context_length.min(MAX_POSITIONS);
        let embed_q = ld.qtensor("token_embd.weight")?;
        let vocab = embed_q.shape().dims()[0];
        // Dequantized to f32, as the gated Qwen3.5 port does: ggml's get_rows dequantizes to f32,
        // and an f16 table would round Q8_0 products the oracle does not round. ~5.6 GB at 31B.
        let embed = Embedding::new(embed_q.dequantize(device)?, h);
        let lm_head = if ld.has("output.weight") {
            ld.qmatmul("output.weight", vocab, h)?
        } else {
            QMatMul::from_weights(Arc::new(embed_q))? // tied embeddings
        };
        let rope_freqs = ld.qtensor("rope_freqs.weight")?.dequantize(device)?.to_vec1::<f32>()?;
        let rope_full = Arc::new(Rope::new(cfg.rope_dimension_count, cfg.rope_freq_base, Some(&rope_freqs), max_pos, device)?);
        let rope_swa = Arc::new(Rope::new(cfg.rope_dimension_count_swa, cfg.rope_freq_base_swa, None, max_pos, device)?);
        let mut layers = Vec::with_capacity(cfg.block_count);
        for i in 0..cfg.block_count {
            let rope = if cfg.is_swa[i] { rope_swa.clone() } else { rope_full.clone() };
            layers.push(ld.layer(&cfg, i, rope)?);
        }
        let norm = ld.norm("output_norm.weight", h, eps)?;
        Ok(Self {
            embed,
            embed_scale: (h as f64).sqrt(),
            layers,
            norm,
            lm_head,
            softcap: cfg.final_logit_softcapping,
            pos: 0,
            device: device.clone(),
            cfg,
        })
    }

    pub fn config(&self) -> &Gemma4Config {
        &self.cfg
    }

    /// Positions consumed so far; the only offset `forward` accepts.
    pub fn position(&self) -> usize {
        self.pos
    }

    fn masks(&self, b: usize, l: usize, offset: usize) -> Result<Masks> {
        if l == 1 {
            return Ok(Masks { full: None, sliding: None });
        }
        let w = self.cfg.sliding_window;
        let prior = offset.min(w);
        let t = |m: Vec<f32>, n: usize| Tensor::from_vec(m, (l, n), &self.device)?.expand((b, 1, l, n));
        Ok(Masks {
            full: Some(t(mask_values(offset, l, 0, offset + l, None), offset + l)?),
            sliding: Some(t(mask_values(offset, l, offset - prior, prior + l, Some(w)), prior + l)?),
        })
    }

    /// Logits of the last position, `[b, vocab]`, softcapped. After an error the state is
    /// undefined: call [`ModelWeights::clear_kv_cache`] or restore a snapshot.
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        if offset != self.pos {
            candle::bail!(
                "forward at position {offset}, but the model's state ends at {}: a snapshot restored \
                 without its position, or a skipped forward",
                self.pos
            )
        }
        let (b, l) = input.dims2()?;
        let masks = self.masks(b, l, offset)?;
        let mut h = (self.embed.forward(input)? * self.embed_scale)?;
        for layer in &mut self.layers {
            h = layer.forward(&h, offset, &masks)?;
        }
        self.pos = offset + l;
        let h = self.norm.forward(&h.narrow(1, l - 1, 1)?)?;
        let logits = self.lm_head.forward(&h)?.squeeze(1)?;
        match self.softcap {
            None => Ok(logits),
            Some(c) => (logits / c)?.tanh()? * c,
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.attn.clear();
        }
        self.pos = 0;
    }

    /// Every layer's state, in layer order. SHARES storage with the model, and the sliding
    /// rings are written in place: deep-copy anything you keep.
    pub fn layer_states(&self) -> Vec<LayerState> {
        self.layers
            .iter()
            .map(|layer| match &layer.attn.kv {
                KvState::Sliding { k, v, write_pos, offset } => {
                    LayerState::Sliding { k: k.clone(), v: v.clone(), write_pos: *write_pos, offset: *offset }
                }
                KvState::Full(c) => LayerState::Full { k: c.k().cloned(), v: c.v().cloned() },
            })
            .collect()
    }

    /// Install `states` AS GIVEN (pass deep copies to keep the snapshot). Validates everything
    /// before touching the model, so a rejected state changes nothing.
    pub fn set_layer_states(&mut self, states: &[LayerState]) -> Result<()> {
        if states.len() != self.layers.len() {
            candle::bail!("state has {} layers, model has {}", states.len(), self.layers.len())
        }
        let window = self.cfg.sliding_window;
        let mut pos: Option<usize> = None;
        for (i, (layer, state)) in self.layers.iter().zip(states).enumerate() {
            let p = match (&layer.attn.kv, state) {
                (KvState::Sliding { .. }, LayerState::Sliding { k, v, write_pos, offset }) => {
                    check_sliding_state(i, k, v, *write_pos, *offset, window)?;
                    *offset
                }
                (KvState::Full(_), LayerState::Full { k, v }) => {
                    if k.is_some() != v.is_some() {
                        candle::bail!("layer {i}: full-attention state is half-set")
                    }
                    state.position()?
                }
                _ => candle::bail!("layer {i}: state kind does not match the model's layer kind"),
            };
            match pos {
                None => pos = Some(p),
                Some(q) if q != p => candle::bail!("layers disagree on the position: layer {i} is at {p}, earlier layers at {q}"),
                Some(_) => {}
            }
        }
        for (layer, state) in self.layers.iter_mut().zip(states) {
            match (&mut layer.attn.kv, state) {
                (KvState::Sliding { k, v, write_pos, offset }, LayerState::Sliding { k: sk, v: sv, write_pos: sw, offset: so }) => {
                    (*k, *v, *write_pos, *offset) = (sk.clone(), sv.clone(), *sw, *so);
                }
                (KvState::Full(c), LayerState::Full { k, v }) => {
                    c.reset();
                    if let (Some(k), Some(v)) = (k, v) {
                        c.append(k, v)?;
                    }
                }
                _ => unreachable!("kinds were checked above"),
            }
        }
        self.pos = pos.unwrap_or(0);
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tiny {
    //! A 3-layer gemma4 GGUF built in memory: layers 0 and 1 sliding (window 4), layer 2 full.
    //! Layer 0 has its own `attn_v`; layers 1 and 2 do not (V = the raw K projection), so both
    //! attention paths run. Values are deterministic, not random, so failures reproduce.
    //!
    //! `attention.head_count_kv` is written as an ARRAY of INT32, matching gguf-py's default for
    //! int lists in the real checkpoint (Task 1's facts file), so the tests exercise the same
    //! integer-width path as the real GGUF, not just the easy U32 scalar path.
    use candle::quantized::gguf_file::{self, Value};
    use candle::quantized::{GgmlDType, QTensor};
    use candle::{DType, Device, Tensor};
    use std::io::Cursor;

    pub const VOCAB: usize = 32;
    pub const HIDDEN: usize = 16;
    pub const FF: usize = 32;
    pub const HEADS: usize = 2;
    pub const KV_HEADS: [usize; 3] = [2, 2, 1];
    pub const HD_SWA: usize = 8;
    pub const HD_FULL: usize = 16;
    pub const WINDOW: usize = 4;
    pub const IS_SWA: [bool; 3] = [true, true, false];
    pub const CONTEXT: usize = 64;

    #[derive(Clone)]
    pub struct Tiny {
        pub arch: &'static str,
        pub expert_count: u32,
        pub per_layer_input: u32,
        pub shared_kv_layers: u32,
        pub extra_tensor: Option<&'static str>,
        pub omit_key: Option<&'static str>,
        pub split_count: Option<u16>,
    }

    impl Default for Tiny {
        fn default() -> Self {
            Self {
                arch: "gemma4",
                expert_count: 0,
                per_layer_input: 0,
                shared_kv_layers: 0,
                extra_tensor: None,
                omit_key: None,
                split_count: None,
            }
        }
    }

    fn q(shape: &[usize], seed: f64, scale: f64, bias: f64) -> QTensor {
        let n: usize = shape.iter().product();
        let t = Tensor::arange(0u32, n as u32, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let t = (((t * (0.37 + seed)).unwrap().sin().unwrap() * scale).unwrap() + bias).unwrap();
        QTensor::quantize(&t.reshape(shape).unwrap(), GgmlDType::F32).unwrap()
    }

    pub fn gguf_bytes(o: &Tiny) -> Vec<u8> {
        let a = o.arch;
        let k = |s: &str| format!("{a}.{s}");
        let u = |x: usize| Value::U32(x as u32);
        let mut md: Vec<(String, Value)> = vec![
            ("general.architecture".into(), Value::String(a.into())),
            (k("block_count"), u(3)),
            (k("context_length"), u(CONTEXT)),
            (k("embedding_length"), u(HIDDEN)),
            (k("feed_forward_length"), u(FF)),
            (k("attention.head_count"), u(HEADS)),
            (
                k("attention.head_count_kv"),
                Value::Array(KV_HEADS.iter().map(|&h| Value::I32(h as i32)).collect()),
            ),
            (k("attention.key_length"), u(HD_FULL)),
            (k("attention.value_length"), u(HD_FULL)),
            (k("attention.key_length_swa"), u(HD_SWA)),
            (k("attention.value_length_swa"), u(HD_SWA)),
            (k("attention.layer_norm_rms_epsilon"), Value::F32(1e-6)),
            (k("attention.sliding_window"), u(WINDOW)),
            (
                k("attention.sliding_window_pattern"),
                Value::Array(IS_SWA.iter().map(|&b| Value::Bool(b)).collect()),
            ),
            (k("attention.shared_kv_layers"), Value::U32(o.shared_kv_layers)),
            (k("embedding_length_per_layer_input"), Value::U32(o.per_layer_input)),
            (k("rope.freq_base"), Value::F32(1_000_000.0)),
            (k("rope.freq_base_swa"), Value::F32(10_000.0)),
            (k("rope.dimension_count"), u(HD_FULL)),
            (k("rope.dimension_count_swa"), u(HD_SWA)),
            (k("final_logit_softcapping"), Value::F32(30.0)),
        ];
        if o.expert_count > 0 {
            md.push((k("expert_count"), Value::U32(o.expert_count)));
        }
        if let Some(n) = o.split_count {
            md.push(("split.count".into(), Value::U16(n)));
        }
        if let Some(omit) = o.omit_key {
            md.retain(|(name, _)| *name != k(omit));
        }
        // Proportional RoPE, as the converter writes it: a quarter of the HD_FULL/2 pairs rotate
        // (factor 1.0), the rest carry 1e30 and are effectively unrotated.
        let rope_freqs = Tensor::new(&[1f32, 1., 1e30, 1e30, 1e30, 1e30, 1e30, 1e30], &Device::Cpu)
            .unwrap();
        let mut ts: Vec<(String, QTensor)> = vec![
            ("token_embd.weight".into(), q(&[VOCAB, HIDDEN], 0.0, 0.5, 0.0)),
            ("output_norm.weight".into(), q(&[HIDDEN], 0.1, 0.1, 1.0)),
            ("rope_freqs.weight".into(), QTensor::quantize(&rope_freqs, GgmlDType::F32).unwrap()),
        ];
        for (i, &swa) in IS_SWA.iter().enumerate() {
            let hd = if swa { HD_SWA } else { HD_FULL };
            let kv = KV_HEADS[i];
            let s = i as f64;
            let b = |n: &str| format!("blk.{i}.{n}.weight");
            ts.push((b("attn_norm"), q(&[HIDDEN], s + 0.2, 0.1, 1.0)));
            ts.push((b("attn_q"), q(&[HEADS * hd, HIDDEN], s + 0.3, 0.4, 0.0)));
            ts.push((b("attn_k"), q(&[kv * hd, HIDDEN], s + 0.4, 0.4, 0.0)));
            if i == 0 {
                ts.push((b("attn_v"), q(&[kv * hd, HIDDEN], s + 0.5, 0.4, 0.0)));
            }
            ts.push((b("attn_output"), q(&[HIDDEN, HEADS * hd], s + 0.6, 0.4, 0.0)));
            ts.push((b("attn_q_norm"), q(&[hd], s + 0.7, 0.1, 1.0)));
            ts.push((b("attn_k_norm"), q(&[hd], s + 0.8, 0.1, 1.0)));
            ts.push((b("post_attention_norm"), q(&[HIDDEN], s + 0.9, 0.1, 1.0)));
            ts.push((b("ffn_norm"), q(&[HIDDEN], s + 1.0, 0.1, 1.0)));
            ts.push((b("ffn_gate"), q(&[FF, HIDDEN], s + 1.1, 0.4, 0.0)));
            ts.push((b("ffn_up"), q(&[FF, HIDDEN], s + 1.2, 0.4, 0.0)));
            ts.push((b("ffn_down"), q(&[HIDDEN, FF], s + 1.3, 0.4, 0.0)));
            ts.push((b("post_ffw_norm"), q(&[HIDDEN], s + 1.4, 0.1, 1.0)));
            ts.push((b("layer_output_scale"), q(&[1], s + 1.5, 0.0, 0.9)));
        }
        if let Some(name) = o.extra_tensor {
            ts.push((name.into(), q(&[4], 9.0, 0.1, 0.0)));
        }
        let md_refs: Vec<(&str, &Value)> = md.iter().map(|(n, v)| (n.as_str(), v)).collect();
        let t_refs: Vec<(&str, &QTensor)> = ts.iter().map(|(n, t)| (n.as_str(), t)).collect();
        let mut c = Cursor::new(Vec::new());
        gguf_file::write(&mut c, &md_refs, &t_refs).unwrap();
        c.into_inner()
    }

    pub fn content(o: &Tiny) -> (gguf_file::Content, Cursor<Vec<u8>>) {
        let mut c = Cursor::new(gguf_bytes(o));
        let ct = gguf_file::Content::read(&mut c).unwrap();
        (ct, c)
    }
}

#[cfg(test)]
mod config_tests {
    use super::tiny::{self, Tiny};
    use super::*;

    fn refusal(o: Tiny) -> String {
        let (ct, _) = tiny::content(&o);
        match check_supported(&ct) {
            Ok(()) => panic!("expected a refusal"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn the_tiny_model_is_supported_and_its_config_reads_back() {
        let (ct, _) = tiny::content(&Tiny::default());
        check_supported(&ct).unwrap();
        let c = Gemma4Config::from_metadata(&ct.metadata).unwrap();
        assert_eq!(c.block_count, 3);
        assert_eq!(c.is_swa, vec![true, true, false]);
        assert_eq!(c.head_count, vec![2, 2, 2]);
        assert_eq!(c.head_count_kv, vec![2, 2, 1]);
        assert_eq!((c.head_dim(0), c.head_dim(2)), (8, 16));
        assert_eq!(c.sliding_window, 4);
        assert_eq!(c.rope_freq_base_swa, 10_000.0);
        assert_eq!(c.final_logit_softcapping, Some(30.0));
    }

    /// Controller ruling: the real GGUF stores `head_count_kv` as an ARRAY of INT32 while scalars
    /// (like `block_count`) are UINT32. `gguf_uint` must read both widths, and reject negatives.
    #[test]
    fn gguf_uint_reads_u32_scalars_and_i32_array_elements_and_rejects_negatives() {
        assert_eq!(gguf_uint(&gguf_file::Value::U32(7)).unwrap(), 7);
        assert_eq!(gguf_uint(&gguf_file::Value::I32(7)).unwrap(), 7);
        assert!(gguf_uint(&gguf_file::Value::I32(-1)).is_err());
    }

    #[test]
    fn a_moe_checkpoint_is_refused() {
        assert!(refusal(Tiny { expert_count: 2, ..Tiny::default() }).contains("MoE"));
    }

    #[test]
    fn a_router_tensor_is_refused_even_without_the_key() {
        let o = Tiny { extra_tensor: Some("blk.0.ffn_gate_inp.weight"), ..Tiny::default() };
        assert!(refusal(o).contains("MoE"));
    }

    #[test]
    fn per_layer_input_embeddings_are_refused() {
        assert!(refusal(Tiny { per_layer_input: 4, ..Tiny::default() }).contains("per-layer"));
    }

    #[test]
    fn shared_kv_layers_are_refused() {
        assert!(refusal(Tiny { shared_kv_layers: 1, ..Tiny::default() }).contains("shared KV"));
    }

    #[test]
    fn a_vision_tower_tensor_is_refused() {
        let o = Tiny { extra_tensor: Some("v.blk.0.attn_q.weight"), ..Tiny::default() };
        assert!(refusal(o).contains("vision"));
    }

    #[test]
    fn a_split_gguf_is_refused() {
        let e = refusal(Tiny { split_count: Some(2), ..Tiny::default() });
        assert!(e.contains("split.count") && e.contains("llama-gguf-split"), "{e}");
    }

    #[test]
    fn another_architecture_is_refused() {
        assert!(refusal(Tiny { arch: "gemma3", ..Tiny::default() }).contains("not a gemma4"));
    }

    #[test]
    fn a_missing_sliding_rope_base_is_an_error_not_a_default() {
        let (ct, _) = tiny::content(&Tiny { omit_key: Some("rope.freq_base_swa"), ..Tiny::default() });
        let e = Gemma4Config::from_metadata(&ct.metadata).unwrap_err().to_string();
        assert!(e.contains("gemma4.rope.freq_base_swa"), "{e}");
    }

    /// The ring code takes positions modulo the window; a zero window must be refused up front.
    #[test]
    fn a_zero_sliding_window_is_refused() {
        let (mut ct, _) = tiny::content(&Tiny::default());
        ct.metadata.insert("gemma4.attention.sliding_window".into(), gguf_file::Value::U32(0));
        let e = Gemma4Config::from_metadata(&ct.metadata).unwrap_err().to_string();
        assert!(e.contains("sliding_window"), "{e}");
    }
}

#[cfg(test)]
mod ring_tests {
    use super::*;
    use candle::{Device, Tensor};

    /// `[1, 1, n, 1]` whose values are the absolute positions `start..start+n`.
    fn positions(start: usize, n: usize) -> Tensor {
        let v: Vec<f32> = (start..start + n).map(|p| p as f32).collect();
        Tensor::from_vec(v, (1, 1, n, 1), &Device::Cpu).unwrap()
    }

    fn values(t: &Tensor) -> Vec<usize> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|&x| x as usize).collect()
    }

    #[test]
    fn the_ring_keeps_the_last_window_positions_each_in_its_modulo_slot() {
        let w = 4;
        for end in [1usize, 3, 4, 5, 7, 8, 9, 13] {
            // up to two positions MORE than the window, to exercise the trim
            let total = end.min(w + 2);
            let (ring, wp) = chronological_to_ring(&positions(end - total, total), end, w).unwrap();
            assert_eq!(wp, end % w, "end={end}");
            let slots = values(&ring);
            assert_eq!(slots.len(), end.min(w), "end={end}");
            if end >= w {
                for (s, p) in slots.iter().enumerate() {
                    assert_eq!(p % w, s, "end={end}: position {p} must sit in slot {}", p % w);
                }
            }
            let back = values(&ring_to_chronological(&ring, wp, w).unwrap());
            let expect: Vec<usize> = (end - end.min(w)..end).collect();
            assert_eq!(back, expect, "end={end}");
        }
    }

    #[test]
    fn a_short_buffer_written_elsewhere_than_its_end_is_rejected() {
        assert!(ring_to_chronological(&positions(0, 2), 1, 4).is_err());
    }

    #[test]
    fn the_sliding_mask_admits_exactly_window_positions_including_self() {
        // query at position 5, window 4: keys 2..=5 visible; 1 (5-1 = 4 back) and 6 (future) not.
        let m = mask_values(5, 1, 0, 7, Some(4));
        let visible: Vec<usize> = (0..7).filter(|&j| m[j] == 0.0).collect();
        assert_eq!(visible, vec![2, 3, 4, 5]);
        let causal = mask_values(5, 1, 0, 7, None);
        assert_eq!((0..7).filter(|&j| causal[j] == 0.0).count(), 6);
    }

    #[test]
    fn check_sliding_state_accepts_only_consistent_positions() {
        let t = |n| Some(positions(0, n));
        assert!(check_sliding_state(0, &None, &None, 0, 0, 4).is_ok());
        assert!(check_sliding_state(0, &t(3), &t(3), 3, 3, 4).is_ok());
        assert!(check_sliding_state(0, &t(4), &t(4), 3, 7, 4).is_ok());
        // the 2026-09-25 Python failure shape: right tensors, wrong position
        assert!(check_sliding_state(0, &t(4), &t(4), 0, 7, 4).is_err());
        assert!(check_sliding_state(0, &t(4), &t(4), 3, 6, 4).is_err());
        assert!(check_sliding_state(0, &t(4), &None, 3, 7, 4).is_err());
        assert!(check_sliding_state(0, &t(3), &t(3), 3, 7, 4).is_err());
    }

    #[test]
    fn deep_copy_shares_no_storage_with_an_in_place_ring_write() {
        let k = positions(0, 4).contiguous().unwrap();
        let s = LayerState::Sliding { k: Some(k.clone()), v: Some(k.clone()), write_pos: 0, offset: 4 };
        let copy = s.deep_copy().unwrap();
        k.slice_set(&positions(99, 1), 2, 0).unwrap();
        let LayerState::Sliding { k: Some(ck), write_pos, offset, .. } = copy else { panic!() };
        assert_eq!(values(&ck), vec![0, 1, 2, 3]);
        assert_eq!((write_pos, offset), (0, 4));
        assert_eq!(s.position().unwrap(), 4);
    }
}

#[cfg(test)]
mod model_tests {
    use super::tiny::{self, Tiny};
    use super::*;
    use candle::{Device, Tensor};

    fn model() -> ModelWeights {
        let (ct, mut c) = tiny::content(&Tiny::default());
        ModelWeights::from_gguf(ct, &mut c, &Device::Cpu).unwrap()
    }

    fn ids(n: usize, salt: u32) -> Vec<u32> {
        (0..n as u32).map(|i| (i * 7 + 3 + salt) % tiny::VOCAB as u32).collect()
    }

    /// Forward `ids` from `start`, `chunk` tokens at a time; the last logits.
    fn run(m: &mut ModelWeights, ids: &[u32], start: usize, chunk: usize) -> Tensor {
        let mut pos = start;
        let mut last = None;
        for c in ids.chunks(chunk) {
            let t = Tensor::new(c, &Device::Cpu).unwrap().unsqueeze(0).unwrap();
            last = Some(m.forward(&t, pos).unwrap());
            pos += c.len();
        }
        last.unwrap()
    }

    fn vec(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        vec(a).iter().zip(vec(b)).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn both_attention_paths_are_loaded() {
        let m = model();
        assert!(m.layers[0].attn.v_proj.is_some());
        assert!(m.layers[1].attn.v_proj.is_none(), "k_eq_v on a sliding layer");
        assert!(m.layers[2].attn.v_proj.is_none(), "k_eq_v on the full layer");
        assert!(matches!(m.layers[2].attn.kv, KvState::Full(_)));
    }

    /// 11 tokens, window 4: the ring wraps twice. Decode (no mask, in-place ring), one-shot
    /// prefill (sliding mask) and 3-token chunks (ring -> chronological -> mask) must agree.
    #[test]
    fn decode_prefill_and_chunked_prefill_agree_across_window_wraps() {
        let x = ids(11, 0);
        let one_shot = run(&mut model(), &x, 0, 11);
        let decode = run(&mut model(), &x, 0, 1);
        let chunked = run(&mut model(), &x, 0, 3);
        assert!(max_diff(&one_shot, &decode) < 1e-4, "{}", max_diff(&one_shot, &decode));
        assert!(max_diff(&one_shot, &chunked) < 1e-4, "{}", max_diff(&one_shot, &chunked));
        let v = vec(&one_shot);
        assert!(v.iter().any(|&a| (a - v[0]).abs() > 1e-3), "degenerate logits prove nothing");
    }

    #[test]
    fn a_restored_snapshot_continues_bit_identically() {
        let mut m = model();
        run(&mut m, &ids(7, 0), 0, 7);
        let snap: Vec<LayerState> = m.layer_states().iter().map(|s| s.deep_copy().unwrap()).collect();
        let LayerState::Sliding { k: Some(k), write_pos, offset, .. } = &snap[0] else { panic!() };
        assert_eq!((k.dim(2).unwrap(), *write_pos, *offset), (tiny::WINDOW, 3, 7), "the ring wrapped");
        let tail = ids(3, 5);
        let a = run(&mut m, &tail, 7, 1);
        let fresh: Vec<LayerState> = snap.iter().map(|s| s.deep_copy().unwrap()).collect();
        m.set_layer_states(&fresh).unwrap();
        assert_eq!(m.position(), 7);
        let b = run(&mut m, &tail, 7, 1);
        assert_eq!(vec(&a), vec(&b));
    }

    /// The cache gate's shape: restore + suffix prefill == the same split computed inline.
    #[test]
    fn restore_then_prefill_equals_the_same_split_inline() {
        let (prefix, suffix) = (ids(7, 0), ids(4, 9));
        let mut inline = model();
        run(&mut inline, &prefix, 0, 7);
        let x = run(&mut inline, &suffix, 7, 4);

        let mut m = model();
        run(&mut m, &prefix, 0, 7);
        let snap: Vec<LayerState> = m.layer_states().iter().map(|s| s.deep_copy().unwrap()).collect();
        run(&mut m, &ids(2, 3), 7, 1); // dirty the state
        m.set_layer_states(&snap).unwrap();
        let y = run(&mut m, &suffix, 7, 4);
        assert_eq!(vec(&x), vec(&y));
    }

    #[test]
    fn a_state_without_its_ring_position_is_refused_and_leaves_the_model_alone() {
        let mut m = model();
        run(&mut m, &ids(7, 0), 0, 7);
        let mut bad: Vec<LayerState> = m.layer_states().iter().map(|s| s.deep_copy().unwrap()).collect();
        if let LayerState::Sliding { write_pos, .. } = &mut bad[1] {
            *write_pos = (*write_pos + 1) % tiny::WINDOW;
        }
        let e = m.set_layer_states(&bad).unwrap_err().to_string();
        assert!(e.contains("inconsistent"), "{e}");
        assert_eq!(m.position(), 7);
    }

    #[test]
    fn layers_that_disagree_on_the_position_are_refused() {
        let mut m = model();
        run(&mut m, &ids(7, 0), 0, 7);
        let mut bad: Vec<LayerState> = m.layer_states().iter().map(|s| s.deep_copy().unwrap()).collect();
        if let LayerState::Full { k, v } = &mut bad[2] {
            *k = Some(k.as_ref().unwrap().narrow(2, 0, 6).unwrap());
            *v = Some(v.as_ref().unwrap().narrow(2, 0, 6).unwrap());
        }
        assert!(m.set_layer_states(&bad).unwrap_err().to_string().contains("disagree"));
    }

    #[test]
    fn a_state_kind_mismatch_is_refused() {
        let mut m = model();
        run(&mut m, &ids(3, 0), 0, 3);
        let mut bad = m.layer_states();
        bad.swap(0, 2);
        assert!(m.set_layer_states(&bad).unwrap_err().to_string().contains("kind"));
    }

    #[test]
    fn a_forward_at_the_wrong_position_is_refused() {
        let mut m = model();
        let t = Tensor::new(&[1u32], &Device::Cpu).unwrap().unsqueeze(0).unwrap();
        assert!(m.forward(&t, 3).unwrap_err().to_string().contains("position"));
    }

    #[test]
    fn positions_beyond_the_rope_table_are_an_error_not_a_panic() {
        let mut m = model();
        run(&mut m, &ids(60, 0), 0, 60);
        let t = Tensor::new(&[1u32, 2, 3, 4, 5], &Device::Cpu).unwrap().unsqueeze(0).unwrap();
        assert!(m.forward(&t, 60).unwrap_err().to_string().contains("beyond"));
    }

    #[test]
    fn softcapped_logits_stay_within_the_cap() {
        let l = run(&mut model(), &ids(5, 0), 0, 5);
        assert!(vec(&l).iter().all(|x| x.abs() <= 30.0));
        // The tiny model's raw logits sit far below 30, so the bound alone cannot tell a missing
        // softcap apart: check the exact `tanh(x / 30) * 30` mapping of the uncapped logits too.
        let mut uncapped = model();
        uncapped.softcap = None;
        let raw = run(&mut uncapped, &ids(5, 0), 0, 5);
        let expect = ((&raw / 30.0).unwrap().tanh().unwrap() * 30.0).unwrap();
        assert!(max_diff(&l, &expect) < 1e-5, "{}", max_diff(&l, &expect));
        assert!(max_diff(&l, &raw) > 1e-4, "the cap must change the logits: {}", max_diff(&l, &raw));
    }

    #[test]
    fn clear_kv_cache_resets_the_position() {
        let mut m = model();
        run(&mut m, &ids(5, 0), 0, 5);
        m.clear_kv_cache();
        assert_eq!(m.position(), 0);
        assert!(m.layer_states().iter().all(|s| s.position().unwrap() == 0));
    }
}
