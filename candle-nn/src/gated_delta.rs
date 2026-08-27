//! Fused GatedDeltaNet recurrence as a candle custom op (Metal only).
//!
//! # Why this file exists in the shape it does — the 6-in/2-out problem
//!
//! candle's richest custom-op trait is [`candle::CustomOp3`]: **three inputs, one output**. This
//! kernel is **six inputs** (`q, k, v, g, beta, state_in`) and **two outputs** (`y` *and*
//! `state_out` — the recurrent state must come back, because the prompt cache snapshots it).
//! mlx does not hit this because `mx.fast.metal_kernel` is natively multi-in/multi-out.
//!
//! The route taken here, and the ones rejected:
//!
//! - **Inputs**: `q, k, v` go through `CustomOp3` normally; `g`, `beta` and `state_in` ride as
//!   FIELDS on the op struct and reach their `MetalStorage` through the public
//!   `Tensor::storage_and_layout()`. No tensor is ever passed BOTH as an argument and as a field:
//!   candle holds a read lock on each argument for the duration of `metal_fwd`.
//! - **Outputs**: ONE flat f32 buffer of `y_elems + state_elems`, split afterwards with
//!   `narrow` + `reshape`. Both are VIEWS on a contiguous 1-D tensor — `reshape` preserves
//!   `start_offset` when the layout is contiguous — so the split costs **zero copies**. That
//!   matters on decode, where `state_out` is as large as the entire rest of the work.
//! - **Rejected**: a candle-core-level entry point (the route the quantized MoE took). It works,
//!   but it is surgery on a crate that needs none of it — the in-tree precedent for a
//!   `candle_metal_kernels::call_*` op is `candle-nn`'s SDPA, and this follows it.
//! - **Rejected**: writing `state_out` back through a mutated input. That would smuggle an output
//!   through an argument candle believes is read-only.

use candle::{Layout, Result, Shape, Tensor};

/// `q, k, v` are the [`candle::CustomOp3`] arguments; these three ride along as fields.
struct GatedDelta {
    g: Tensor,
    beta: Tensor,
    state_in: Tensor,
    /// Element count of `y`, i.e. the offset at which `state_out` starts in the packed output.
    y_elems: usize,
}

#[cfg(feature = "metal")]
fn metal_storage_of(t: &Tensor) -> Result<(std::sync::RwLockReadGuard<'_, candle::Storage>, &Layout)>
{
    let (s, l) = t.storage_and_layout();
    match &*s {
        candle::Storage::Metal(_) => {}
        _ => candle::bail!("gated_delta: expected a metal tensor"),
    }
    if !l.is_contiguous() {
        candle::bail!("gated_delta: inputs must be contiguous");
    }
    Ok((s, l))
}

impl candle::CustomOp3 for GatedDelta {
    fn name(&self) -> &'static str {
        "metal-gated-delta"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &Layout,
        _s2: &candle::CpuStorage,
        _l2: &Layout,
        _s3: &candle::CpuStorage,
        _l3: &Layout,
    ) -> Result<(candle::CpuStorage, Shape)> {
        // No CPU arm on purpose. A fallback here would make a pricing round silently measure the
        // reference implementation twice and report the difference as a speedup.
        candle::bail!("gated_delta has no cpu impl; use the chunked scan on CPU")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        q: &candle::MetalStorage,
        q_l: &Layout,
        k: &candle::MetalStorage,
        k_l: &Layout,
        v: &candle::MetalStorage,
        v_l: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::DType;

        let device = q.device();

        for (name, l) in [("q", q_l), ("k", k_l), ("v", v_l)] {
            if !l.is_contiguous() {
                candle::bail!("gated_delta: {name} must be contiguous");
            }
        }
        for (name, dt) in [("q", q.dtype()), ("k", k.dtype()), ("v", v.dtype())] {
            if dt != DType::F32 {
                candle::bail!("gated_delta: {name} must be f32, got {dt:?}");
            }
        }

        let (b, t, h, dk) = q_l.shape().dims4()?;
        if k_l.shape().dims4()? != (b, t, h, dk) {
            candle::bail!("gated_delta: k {:?} must match q {:?}", k_l.dims(), q_l.dims());
        }
        let (vb, vt, vh, dv) = v_l.shape().dims4()?;
        if (vb, vt, vh) != (b, t, h) {
            candle::bail!("gated_delta: v {:?} disagrees with q {:?}", v_l.dims(), q_l.dims());
        }

        let (g_s, g_l) = metal_storage_of(&self.g)?;
        let (beta_s, beta_l) = metal_storage_of(&self.beta)?;
        let (st_s, st_l) = metal_storage_of(&self.state_in)?;
        if g_l.shape().dims3()? != (b, t, h) {
            candle::bail!("gated_delta: g {:?}, expected [{b}, {t}, {h}]", g_l.dims());
        }
        if beta_l.shape().dims3()? != (b, t, h) {
            candle::bail!("gated_delta: beta {:?}, expected [{b}, {t}, {h}]", beta_l.dims());
        }
        if st_l.shape().dims4()? != (b, h, dk, dv) {
            candle::bail!(
                "gated_delta: state_in {:?}, expected [{b}, {h}, {dk}, {dv}]",
                st_l.dims()
            );
        }

        let (g_m, beta_m, st_m) = match (&*g_s, &*beta_s, &*st_s) {
            (
                candle::Storage::Metal(g),
                candle::Storage::Metal(beta),
                candle::Storage::Metal(st),
            ) => (g, beta, st),
            _ => candle::bail!("gated_delta: g/beta/state_in must be metal tensors"),
        };
        for (name, dt) in [
            ("g", g_m.dtype()),
            ("beta", beta_m.dtype()),
            ("state_in", st_m.dtype()),
        ] {
            if dt != DType::F32 {
                candle::bail!("gated_delta: {name} must be f32, got {dt:?}");
            }
        }

        let y_elems = b * t * h * dv;
        let state_elems = b * h * dk * dv;
        if y_elems != self.y_elems {
            candle::bail!("gated_delta: internal y_elems mismatch");
        }
        let out_shape = Shape::from_dims(&[y_elems + state_elems]);

        let output = device
            .new_buffer_builder()
            .with_size_for(y_elems + state_elems, DType::F32)
            .with_label("gated_delta_packed")
            .build()?;

        const F32: usize = std::mem::size_of::<f32>();
        let off = |l: &Layout| l.start_offset() * F32;

        let encoder = device.command_encoder()?;
        candle_metal_kernels::call_gated_delta(
            device.device(),
            &encoder,
            device.kernels(),
            b,
            t,
            h,
            dk,
            dv,
            (q.buffer(), off(q_l)),
            (k.buffer(), off(k_l)),
            (v.buffer(), off(v_l)),
            (g_m.buffer(), off(g_l)),
            (beta_m.buffer(), off(beta_l)),
            (st_m.buffer(), off(st_l)),
            (&output, 0),
            (&output, y_elems * F32),
        )
        .map_err(candle::Error::wrap)?;

        let storage = candle::MetalStorage::new(
            output,
            device.clone(),
            y_elems + state_elems,
            DType::F32,
        );
        Ok((storage, out_shape))
    }
}

/// One fused launch of the gated delta recurrence over the WHOLE `t` axis.
///
/// Inputs (all f32, `Hk == Hv == h`: the caller must pre-expand q/k to the v-head count, because
/// this kernel's head mapping is deliberately the identity):
///   `q`, `k`:    `[b, t, h, dk]`, already l2-normalised and q already scaled
///   `v`:         `[b, t, h, dv]`
///   `g`:         `[b, t, h]` — the DECAY ITSELF, i.e. `exp(log_g)`, not the log-decay
///   `beta`:      `[b, t, h]`
///   `state_in`:  `[b, h, dk, dv]`
///
/// Returns `(y [b, t, h, dv], state_out [b, h, dk, dv])`.
pub fn gated_delta(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state_in: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (b, t, h, _dk) = q.dims4()?;
    let dv = v.dim(3)?;
    let dk = state_in.dim(2)?;
    let y_elems = b * t * h * dv;
    let state_elems = b * h * dk * dv;

    // `contiguous()` is a no-op clone when the layout already is contiguous, so this is a guard,
    // not a copy on the hot path.
    let (q, k, v) = (q.contiguous()?, k.contiguous()?, v.contiguous()?);
    let op = GatedDelta {
        g: g.contiguous()?,
        beta: beta.contiguous()?,
        state_in: state_in.contiguous()?,
        y_elems,
    };
    let packed = q.apply_op3_no_bwd(&k, &v, &op)?;

    // Zero-copy split: `narrow` on a contiguous 1-D tensor is a view, and `reshape` keeps the
    // start offset when the layout is contiguous.
    let y = packed.narrow(0, 0, y_elems)?.reshape((b, t, h, dv))?;
    let state_out = packed
        .narrow(0, y_elems, state_elems)?
        .reshape((b, h, dk, dv))?;
    Ok((y, state_out))
}
