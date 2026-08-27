//! Unit tests for the fused GatedDeltaNet Metal kernel (`candle_nn::gated_delta::gated_delta`).
//!
//! WHY THIS FILE EXISTS: the kernel shipped with NO unit test of any kind. Every gate that covered
//! it ran the prompt cache OFF, which means:
//!
//!   * fresh prefill exercises (`t > 1`, `state_in == 0`)
//!   * decode exercises (`t == 1`, `state_in != 0`)
//!   * **nothing** exercised (`t > 1`, `state_in != 0`)
//!
//! and (`t > 1`, `state_in != 0`) is exactly what restore-then-suffix-prefill does. These tests
//! pin that combination, and pin the property the prompt cache actually depends on: splitting a
//! sequence and carrying the state must give the SAME answer as one long launch.
#![cfg(feature = "metal")]

use candle::{DType, Device, Result, Tensor, D};
use candle_transformers::models::qwen3_5_linear_attn_scan::gated_delta_rule_chunked;

const DK: usize = 128;
const DV: usize = 128;

/// Inputs shaped the way `torch_recurrent_gated_delta_rule` shapes them before the call:
/// q/k l2-normalised, q scaled by 1/sqrt(dk), beta in (0, 1), log_g <= 0.
struct Inputs {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    log_g: Tensor,
    beta: Tensor,
}

fn l2_norm(xs: &Tensor) -> Result<Tensor> {
    let norm = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
    xs.broadcast_div(&norm)
}

impl Inputs {
    fn random(b: usize, t: usize, h: usize, dev: &Device) -> Result<Self> {
        let q = l2_norm(&Tensor::randn(0f32, 1f32, (b, t, h, DK), dev)?)?;
        let q = (q * (1.0 / (DK as f64).sqrt()))?;
        let k = l2_norm(&Tensor::randn(0f32, 1f32, (b, t, h, DK), dev)?)?;
        let v = Tensor::randn(0f32, 1f32, (b, t, h, DV), dev)?;
        // beta = sigmoid(x) in (0, 1); log_g = -softplus(x) <= 0, the decay's LOG.
        let beta = candle_nn::ops::sigmoid(&Tensor::randn(0f32, 1f32, (b, t, h), dev)?)?;
        let log_g = Tensor::randn(0f32, 1f32, (b, t, h), dev)?;
        let log_g = ((log_g.exp()? + 1.0)?.log()? * -0.05)?;
        Ok(Self {
            q,
            k,
            v,
            log_g,
            beta,
        })
    }

    /// `[from, from + n)` along the time axis.
    fn slice(&self, from: usize, n: usize) -> Result<Self> {
        Ok(Self {
            q: self.q.narrow(1, from, n)?.contiguous()?,
            k: self.k.narrow(1, from, n)?.contiguous()?,
            v: self.v.narrow(1, from, n)?.contiguous()?,
            log_g: self.log_g.narrow(1, from, n)?.contiguous()?,
            beta: self.beta.narrow(1, from, n)?.contiguous()?,
        })
    }

    /// The fused kernel. Takes the DECAY (exp of log_g), as the model's call site does.
    fn kernel(&self, state: &Tensor) -> Result<(Tensor, Tensor)> {
        candle_nn::gated_delta::gated_delta(
            &self.q,
            &self.k,
            &self.v,
            &self.log_g.exp()?,
            &self.beta,
            state,
        )
    }

    /// The reference chunked scan. Takes the LOG-decay and updates `state` in place.
    fn chunked(&self, state: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut state = state.clone();
        let y = gated_delta_rule_chunked(
            &self.q,
            &self.k,
            &self.v,
            &self.log_g,
            &self.beta,
            &mut state,
        )?;
        Ok((y, state))
    }
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a - b)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()
}

fn device() -> Result<Device> {
    Device::new_metal(0)
}

/// A state that is emphatically NOT zero, produced the way the model produces one: by running the
/// recurrence over some earlier tokens. Using a random tensor instead would be a weaker test —
/// the real restore path feeds back a state the kernel itself wrote.
fn warm_state(b: usize, h: usize, dev: &Device) -> Result<Tensor> {
    let zero = Tensor::zeros((b, h, DK, DV), DType::F32, dev)?;
    let warm = Inputs::random(b, 8, h, dev)?;
    let (_, state) = warm.chunked(&zero)?;
    Ok(state)
}

/// CONTROL: the combination the existing end-to-end gates already cover — `t > 1` from a ZERO
/// state. If this ever fails, the failure is not about state carry and the tests below say nothing.
#[test]
fn kernel_matches_chunked_from_zero_state() -> Result<()> {
    let dev = device()?;
    let (b, h, t) = (1, 2, 16);
    let inp = Inputs::random(b, t, h, &dev)?;
    let zero = Tensor::zeros((b, h, DK, DV), DType::F32, &dev)?;

    let (y_kernel, s_kernel) = inp.kernel(&zero)?;
    let (y_chunked, s_chunked) = inp.chunked(&zero)?;

    let dy = max_abs_diff(&y_kernel, &y_chunked)?;
    let ds = max_abs_diff(&s_kernel, &s_chunked)?;
    assert!(dy < 1e-4, "y differs by {dy} from a zero state");
    assert!(ds < 1e-4, "state differs by {ds} from a zero state");
    Ok(())
}

/// CONTROL: `t == 1` from a non-zero state — the decode path, which the llama.cpp oracle covers.
#[test]
fn kernel_matches_chunked_single_step_from_warm_state() -> Result<()> {
    let dev = device()?;
    let (b, h) = (1, 2);
    let state = warm_state(b, h, &dev)?;
    let inp = Inputs::random(b, 1, h, &dev)?;

    let (y_kernel, s_kernel) = inp.kernel(&state)?;
    let (y_chunked, s_chunked) = inp.chunked(&state)?;

    let dy = max_abs_diff(&y_kernel, &y_chunked)?;
    let ds = max_abs_diff(&s_kernel, &s_chunked)?;
    assert!(dy < 1e-4, "y differs by {dy} at t == 1");
    assert!(ds < 1e-4, "state differs by {ds} at t == 1");
    Ok(())
}

/// THE UNCOVERED COMBINATION: `t > 1` AND `state_in != 0`.
///
/// This is what restore-then-suffix-prefill does and what nothing has ever exercised.
#[test]
fn kernel_matches_chunked_multi_step_from_warm_state() -> Result<()> {
    let dev = device()?;
    let (b, h, t) = (1, 2, 16);
    let state = warm_state(b, h, &dev)?;
    let inp = Inputs::random(b, t, h, &dev)?;

    let (y_kernel, s_kernel) = inp.kernel(&state)?;
    let (y_chunked, s_chunked) = inp.chunked(&state)?;

    let dy = max_abs_diff(&y_kernel, &y_chunked)?;
    let ds = max_abs_diff(&s_kernel, &s_chunked)?;
    assert!(dy < 1e-4, "y differs by {dy} at t > 1 from a warm state");
    assert!(ds < 1e-4, "state differs by {ds} at t > 1 from a warm state");
    Ok(())
}

/// THE PROPERTY THE PROMPT CACHE DEPENDS ON: one launch over T tokens must equal two launches
/// over T/2 tokens each with the state carried between them.
///
/// The kernel is a strictly sequential recurrence held in registers, so this is not an
/// approximate property — it should hold BIT-EXACTLY, which is why the tolerance is 0.
#[test]
fn kernel_split_with_carried_state_equals_one_launch() -> Result<()> {
    let dev = device()?;
    let (b, h, t) = (1, 2, 16);
    let inp = Inputs::random(b, t, h, &dev)?;
    let zero = Tensor::zeros((b, h, DK, DV), DType::F32, &dev)?;

    let (y_whole, s_whole) = inp.kernel(&zero)?;

    let (y_a, s_a) = inp.slice(0, t / 2)?.kernel(&zero)?;
    let (y_b, s_b) = inp.slice(t / 2, t / 2)?.kernel(&s_a)?;
    let y_split = Tensor::cat(&[&y_a, &y_b], 1)?;

    let dy = max_abs_diff(&y_whole, &y_split)?;
    let ds = max_abs_diff(&s_whole, &s_b)?;
    assert_eq!(dy, 0.0, "y differs by {dy} between one launch and two");
    assert_eq!(ds, 0.0, "state differs by {ds} between one launch and two");
    Ok(())
}

/// The same split property, but with the second launch's state having gone through a DEEP COPY
/// first — which is what `PromptSnapshot::restore` does. If the plain split passes and this one
/// fails, the defect is in how the state survives being copied, not in the recurrence.
#[test]
fn kernel_split_survives_a_deep_copied_state() -> Result<()> {
    let dev = device()?;
    let (b, h, t) = (1, 2, 16);
    let inp = Inputs::random(b, t, h, &dev)?;
    let zero = Tensor::zeros((b, h, DK, DV), DType::F32, &dev)?;

    let (y_whole, s_whole) = inp.kernel(&zero)?;

    let (y_a, s_a) = inp.slice(0, t / 2)?.kernel(&zero)?;
    // A fresh allocation holding the same bytes, as a restored snapshot would be.
    let s_a_copy = Tensor::from_vec(
        s_a.flatten_all()?.to_vec1::<f32>()?,
        (b, h, DK, DV),
        &dev,
    )?;
    let (y_b, s_b) = inp.slice(t / 2, t / 2)?.kernel(&s_a_copy)?;
    let y_split = Tensor::cat(&[&y_a, &y_b], 1)?;

    let dy = max_abs_diff(&y_whole, &y_split)?;
    let ds = max_abs_diff(&s_whole, &s_b)?;
    assert_eq!(dy, 0.0, "y differs by {dy} with a deep-copied carried state");
    assert_eq!(ds, 0.0, "state differs by {ds} with a deep-copied carried state");
    Ok(())
}

/// The real checkpoint's shape: `h == 32` heads, and a 120-token suffix carried onto a state warmed
/// over 5184 tokens' worth of recurrence — i.e. the exact geometry `equivalence_cache.sh` fails on.
#[test]
fn kernel_at_the_real_shape_split_and_warm() -> Result<()> {
    let dev = device()?;
    let (b, h) = (1, 32);
    let zero = Tensor::zeros((b, h, DK, DV), DType::F32, &dev)?;

    // Split invariance at the real head count, prefix 256 + suffix 120.
    let inp = Inputs::random(b, 376, h, &dev)?;
    let (y_whole, s_whole) = inp.kernel(&zero)?;
    let (y_a, s_a) = inp.slice(0, 256)?.kernel(&zero)?;
    let (y_b, s_b) = inp.slice(256, 120)?.kernel(&s_a)?;
    let y_split = Tensor::cat(&[&y_a, &y_b], 1)?;
    assert_eq!(
        max_abs_diff(&y_whole, &y_split)?,
        0.0,
        "y differs between one launch and a 256+120 split at h = 32"
    );
    assert_eq!(
        max_abs_diff(&s_whole, &s_b)?,
        0.0,
        "state differs between one launch and a 256+120 split at h = 32"
    );

    // And the same suffix against the chunked reference from that warm state.
    let (y_ref, s_ref) = inp.slice(256, 120)?.chunked(&s_a)?;
    let dy = max_abs_diff(&y_b, &y_ref)?;
    let ds = max_abs_diff(&s_b, &s_ref)?;
    assert!(dy < 1e-4, "y differs by {dy} from chunked on the warm suffix");
    assert!(ds < 1e-4, "state differs by {ds} from chunked on the warm suffix");
    Ok(())
}

/// The restore path's mechanics, reproduced at the REAL offsets.
///
/// Under `DeltaMode::Kernel` the returned `state_out` is a `narrow` into the packed `[y | state]`
/// buffer, so its `start_offset` is `b * t * h * dv` elements — 21 233 664 elements (84.9 MB) after
/// a 5184-token prefix. `PromptSnapshot::capture`/`restore` then run `Tensor::copy()` on that view.
/// If the copy or the offset dispatch dropped the offset, the restored state would be garbage.
#[test]
fn restored_state_survives_copy_at_the_real_offset() -> Result<()> {
    let dev = device()?;
    let (b, h) = (1, 32);
    let (t_prefix, t_suffix) = (5184, 120);
    let zero = Tensor::zeros((b, h, DK, DV), DType::F32, &dev)?;

    let prefix = Inputs::random(b, t_prefix, h, &dev)?;
    let (_, s_prefix) = prefix.kernel(&zero)?;
    // 21 233 664 elements into the packed buffer — the offset the model really carries.
    assert_eq!(
        s_prefix.layout().start_offset(),
        b * t_prefix * h * DV,
        "state_out is expected to be a view at the end of the packed buffer"
    );
    // What capture() then restore() do: Tensor::copy(), twice.
    let s_restored = s_prefix.copy()?.copy()?;
    assert_eq!(
        max_abs_diff(&s_prefix, &s_restored)?,
        0.0,
        "deep-copying the state view changed it"
    );

    let suffix = Inputs::random(b, t_suffix, h, &dev)?;
    let (y_live, s_live) = suffix.kernel(&s_prefix)?;
    let (y_restored, s_restored_out) = suffix.kernel(&s_restored)?;
    assert_eq!(
        max_abs_diff(&y_live, &y_restored)?,
        0.0,
        "y differs between the live state view and its deep copy"
    );
    assert_eq!(
        max_abs_diff(&s_live, &s_restored_out)?,
        0.0,
        "state differs between the live state view and its deep copy"
    );
    Ok(())
}
