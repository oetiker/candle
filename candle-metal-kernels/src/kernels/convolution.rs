use crate::linear_split;
use crate::utils::{BufferOffset, EncoderProvider};
use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};
use objc2_metal::MTLSize;

#[allow(clippy::too_many_arguments)]
pub fn call_im2col1d_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (k_size, stride, padding, dilation): (usize, usize, usize, usize),
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let l_out = (shape[2] + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;
    let dst_el = shape[0] * l_out * shape[1] * k_size;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "im2col1d {name} dst_el={dst_el}");
    set_params!(
        encoder,
        (
            dst_el,
            l_out,
            k_size,
            stride,
            padding,
            dilation,
            shape,
            strides,
            &input,
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_col2im1d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    k_size: usize,
    stride: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let l_in = shape[1];
    let c_out = shape[2];
    let l_out = (l_in - 1) * stride + k_size;
    let dst_el = shape[0] * c_out * l_out;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "col2im1d {name} dst_el={dst_el}");
    set_params!(
        encoder,
        (
            dst_el,
            l_out,
            l_in,
            c_out,
            k_size,
            stride,
            &input,
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_im2col_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (h_k, w_k, stride, padding, dilation): (usize, usize, usize, usize, usize),
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;

    let h = shape[2];
    let w = shape[3];
    let h_out = (h + 2 * padding - dilation * (h_k - 1) - 1) / stride + 1;
    let w_out = (w + 2 * padding - dilation * (w_k - 1) - 1) / stride + 1;

    let dst_el = shape[0] * h_out * w_out * shape[1] * h_k * w_k;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "im2col {name} dst_el={dst_el}");
    set_params!(
        encoder,
        (
            dst_el,
            h_out,
            w_out,
            h_k,
            w_k,
            stride,
            padding,
            dilation,
            shape,
            strides,
            &input,
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_upsample_nearest_2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let dst_el = out_w * out_h * shape[0] * shape[1];
    let scale_w = shape[2] as f32 / out_w as f32;
    let scale_h = shape[3] as f32 / out_h as f32;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "upsample_nearest2d {name} {out_w}x{out_h}");
    set_params!(
        encoder,
        (
            out_w,
            out_h,
            scale_w,
            scale_h,
            shape,
            strides,
            &input,
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_upsample_bilinear_2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    align_corners: bool,
    scale_h: Option<f64>,
    scale_w: Option<f64>,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let dst_el = out_w * out_h * shape[0] * shape[1];

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "upsample_bilinear2d {name} {out_w}x{out_h}");

    set_params!(
        encoder,
        (
            out_w,
            out_h,
            align_corners,
            scale_h.is_some(),
            scale_h.unwrap_or(0.0) as f32,
            scale_w.is_some(),
            scale_w.unwrap_or(0.0) as f32,
            shape,
            strides,
            &input,
            Output::new(output)
        )
    );

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_pool2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    w_k: usize,
    h_k: usize,
    w_stride: usize,
    h_stride: usize,
    input: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let dst_el = out_w * out_h * shape[0] * shape[1];
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "pool2d {name} {out_w}x{out_h} k={w_k}x{h_k}");
    set_params!(
        encoder,
        (
            w_k,
            h_k,
            w_stride,
            h_stride,
            shape,
            strides,
            input,
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_conv_transpose1d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    dilation: usize,
    stride: usize,
    padding: usize,
    out_padding: usize,
    c_out: usize,
    l_out: usize,
    b_size: usize,
    src_shape: &[usize],
    src_strides: &[usize],
    kernel_shape: &[usize],
    kernel_strides: &[usize],
    input: &Buffer,
    input_offset: usize,
    kernel: &Buffer,
    kernel_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let dst_el = c_out * l_out * b_size;
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "conv_transpose1d {name} c_out={c_out} l_out={l_out} b={b_size}"
    );
    set_params!(
        encoder,
        (
            l_out,
            stride,
            padding,
            out_padding,
            dilation,
            src_shape,
            src_strides,
            kernel_shape,
            kernel_strides,
            (input, input_offset),
            (kernel, kernel_offset),
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Everything `call_conv2d_grouped_tiled` needs that is not a buffer. Grouped: the input has
/// `groups * c_in_pg` channels and the output `groups * c_out_pg`.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dGroupedCfg {
    pub b_size: usize,
    pub groups: usize,
    pub c_in_pg: usize,
    pub c_out_pg: usize,
    pub h_in: usize,
    pub w_in: usize,
    pub h_out: usize,
    pub w_out: usize,
    pub k_h: usize,
    pub k_w: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl Conv2dGroupedCfg {
    /// The number of output elements: `b * groups * c_out_pg * h_out * w_out`. The single
    /// source of truth for this product — the caller (`MetalStorage::conv2d_grouped`) uses it
    /// to size the output buffer, so it must never drift from what the tiled kernel actually
    /// dispatches.
    pub fn dst_el(&self) -> usize {
        self.b_size * self.groups * self.c_out_pg * self.h_out * self.w_out
    }
}

/// One instantiated `conv2d_grouped_tiled` pipeline: its entry point plus the tile constants the
/// shader was compiled with. The dispatch geometry is a function of these, so they must be quoted
/// from the same place the entry point is chosen — a `tile_t` that disagrees with the compiled
/// `TILE_T` would silently compute the wrong columns.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dGroupedTiledVariant {
    /// Shader entry point, e.g. `conv2d_grouped_tiled_f32_t224_c4_r8x4`.
    name: &'static str,
    /// `TILE_T`: output columns per threadgroup.
    tile_t: usize,
    /// `CI_CHUNK`: input channels staged per pass. `c_in_pg` must be a multiple of it.
    ci_chunk: usize,
    /// `(TILE_T / T_REG) * (32 / CO_REG)`: threads per threadgroup.
    threads: usize,
}

impl Conv2dGroupedTiledVariant {
    /// The one shader instantiated in `conv.metal`: `conv2d_grouped_tiled<float, 224, 4, 8, 4>`.
    /// Fields are private and only reachable through this named constant (or others like it, were
    /// more instantiated) so the geometry quoted to `call_conv2d_grouped_tiled` can never drift
    /// from what the shader was actually compiled with -- a mismatched `tile_t` would silently
    /// compute the wrong columns, per the module doc comment above.
    pub const T224_C4_R8X4: Self = Self {
        name: "conv2d_grouped_tiled_f32_t224_c4_r8x4",
        tile_t: 224,
        ci_chunk: 4,
        threads: 224,
    };

    /// `CI_CHUNK`: input channels staged per pass. Callers use this to check `c_in_pg % ci_chunk
    /// == 0` before dispatching.
    pub const fn ci_chunk(&self) -> usize {
        self.ci_chunk
    }
}

/// The tiled grouped conv2d. Only valid for the restricted case the shader documents:
/// `k_h == k_w == 3`, `stride == 1`, `dilation == 1`, `padding == 1`, `c_out_pg == 32`, and
/// `c_in_pg` a multiple of `variant.ci_chunk`.
#[allow(clippy::too_many_arguments)]
pub fn call_conv2d_grouped_tiled(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    variant: Conv2dGroupedTiledVariant,
    cfg: Conv2dGroupedCfg,
    input: BufferOffset,
    weight: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, variant.name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "conv2d_grouped_tiled {}", variant.name);
    set_params!(
        encoder,
        (
            cfg.groups,
            cfg.c_in_pg,
            cfg.h_in,
            cfg.w_in,
            &input,
            &weight,
            Output::new(output)
        )
    );
    let grid_dims = MTLSize {
        width: cfg.w_out.div_ceil(variant.tile_t),
        height: cfg.h_out,
        depth: cfg.b_size * cfg.groups,
    };
    let group_dims = MTLSize {
        width: variant.threads,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

pub struct CallConvTranspose2dCfg<'a> {
    pub dilation: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub c_out: usize,
    pub out_w: usize,
    pub out_h: usize,
    pub b_size: usize,
    pub input_dims: &'a [usize],
    pub input_stride: &'a [usize],
    pub kernel_dims: &'a [usize],
    pub kernel_stride: &'a [usize],
    pub input_offset: usize,
    pub kernel_offset: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn call_conv_transpose2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    cfg: CallConvTranspose2dCfg,
    input: &Buffer,
    kernel: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let dst_el = cfg.c_out * cfg.out_w * cfg.out_h * cfg.b_size;
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "conv_transpose2d {name} c_out={} {}x{} b={}",
        cfg.c_out,
        cfg.out_w,
        cfg.out_h,
        cfg.b_size
    );
    set_params!(
        encoder,
        (
            cfg.out_w,
            cfg.out_h,
            cfg.stride,
            cfg.padding,
            cfg.output_padding,
            cfg.dilation,
            cfg.input_dims,
            cfg.input_stride,
            cfg.kernel_dims,
            cfg.kernel_stride,
            (input, cfg.input_offset),
            (kernel, cfg.kernel_offset),
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
