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

use candle::quantized::gguf_file;
use candle::Result;
use std::collections::HashMap;

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
            sliding_window: md_usize(m, "attention.sliding_window")?,
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
}
