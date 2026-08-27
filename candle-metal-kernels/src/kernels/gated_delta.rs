use crate::utils::Input;
use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, EncoderProvider, Kernels,
    MetalKernelError, Output, Source,
};
use objc2_metal::MTLSize;

/// One launch of the fused GatedDeltaNet recurrence — see `metal_src/gated_delta.metal`.
///
/// Shapes (all f32, all contiguous, `Hk == Hv == h` because the caller pre-expands q/k):
///   q, k:      [b, t, h, dk]
///   v:         [b, t, h, dv]
///   g:         [b, t, h]      the DECAY, already exponentiated (not the log-decay)
///   beta:      [b, t, h]
///   state_in:  [b, h, dk, dv]
///   y:         [b, t, h, dv]
///   state_out: [b, h, dk, dv]
///
/// `y` and `state_out` may live in the SAME buffer at different offsets; the kernel never reads
/// one through the other.
///
/// `dk`/`dv` are compile-time template parameters on the Metal side, so only instantiated shapes
/// work. Anything else is a hard error, never a silent fallback to a slower path.
#[allow(clippy::too_many_arguments)]
pub fn call_gated_delta(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    b: usize,
    t: usize,
    h: usize,
    dk: usize,
    dv: usize,
    q: (&Buffer, usize),
    k: (&Buffer, usize),
    v: (&Buffer, usize),
    g: (&Buffer, usize),
    beta: (&Buffer, usize),
    state_in: (&Buffer, usize),
    y: (&Buffer, usize),
    state_out: (&Buffer, usize),
) -> Result<(), MetalKernelError> {
    // The state row is spread across exactly one 32-wide SIMD group, so this is not merely the
    // instantiation list talking: a dk that is not a multiple of 32 would silently drop lanes.
    let name = match (dk, dv) {
        (128, 128) => "gated_delta_f32_dk128_dv128",
        _ => {
            return Err(MetalKernelError::LoadFunctionError(format!(
                "gated_delta: no kernel instantiated for dk={dk} dv={dv} (have dk=128 dv=128). \
                 Add an instantiate_gated_delta line rather than falling back."
            )))
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::GatedDelta, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gated_delta b={b} t={t} h={h} dk={dk} dv={dv}");

    let t_i32 = t as i32;
    let h_i32 = h as i32;
    set_params!(
        encoder,
        (
            Input::with_offset(q.0, q.1),
            Input::with_offset(k.0, k.1),
            Input::with_offset(v.0, v.1),
            Input::with_offset(g.0, g.1),
            Input::with_offset(beta.0, beta.1),
            Input::with_offset(state_in.0, state_in.1),
            Output::with_offset(y.0, y.1),
            Output::with_offset(state_out.0, state_out.1),
            t_i32,
            h_i32
        )
    );

    // mlx's launch geometry, verbatim: grid (32, Dv, B*Hv) THREADS, threadgroup (32, 4, 1).
    // x must be 32 so `simd_sum` reduces over the whole state row; y is the state row index.
    let grid = MTLSize {
        width: 32,
        height: dv,
        depth: b * h,
    };
    let group = MTLSize {
        width: 32,
        height: 4,
        depth: 1,
    };
    encoder.dispatch_threads(grid, group);
    Ok(())
}
