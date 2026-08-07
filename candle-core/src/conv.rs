//! 1D and 2D Convolutions
//!
use crate::{op::BackpropOp, op::Op, Error, Result, Tensor};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv1D {
    pub(crate) b_size: usize,
    // Maybe we should have a version without l_in as this bit depends on the input and not only on
    // the weights.
    pub(crate) l_in: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) k_size: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
    pub(crate) cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl ParamsConv1D {
    pub(crate) fn l_out(&self) -> usize {
        (self.l_in + 2 * self.padding - self.dilation * (self.k_size - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        let l_out = self.l_out();
        vec![self.b_size, self.c_out, l_out]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConvTranspose1D {
    pub(crate) b_size: usize,
    pub(crate) l_in: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) k_size: usize,
    pub(crate) padding: usize,
    pub(crate) output_padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

impl ParamsConvTranspose1D {
    pub(crate) fn l_out(&self) -> usize {
        (self.l_in - 1) * self.stride - 2 * self.padding
            + self.dilation * (self.k_size - 1)
            + self.output_padding
            + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        let l_out = self.l_out();
        vec![self.b_size, self.c_out, l_out]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudnnFwdAlgo {
    ImplicitGemm,
    ImplicitPrecompGemm,
    Gemm,
    Direct,
    Fft,
    FftTiling,
    Winograd,
    WinogradNonFused,
    Count,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv2D {
    pub(crate) b_size: usize,
    pub(crate) i_h: usize,
    pub(crate) i_w: usize,
    pub(crate) k_h: usize,
    pub(crate) k_w: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl ParamsConv2D {
    pub(crate) fn out_h(&self) -> usize {
        (self.i_h + 2 * self.padding - self.dilation * (self.k_h - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_w(&self) -> usize {
        (self.i_w + 2 * self.padding - self.dilation * (self.k_w - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        vec![self.b_size, self.c_out, self.out_h(), self.out_w()]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConvTranspose2D {
    pub(crate) b_size: usize,
    pub(crate) i_h: usize,
    pub(crate) i_w: usize,
    pub(crate) k_h: usize,
    pub(crate) k_w: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) padding: usize,
    pub(crate) output_padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

impl ParamsConvTranspose2D {
    pub(crate) fn out_h(&self) -> usize {
        (self.i_h - 1) * self.stride + self.dilation * (self.k_h - 1) + self.output_padding + 1
            - 2 * self.padding
    }

    pub(crate) fn out_w(&self) -> usize {
        (self.i_w - 1) * self.stride + self.dilation * (self.k_w - 1) + self.output_padding + 1
            - 2 * self.padding
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        vec![self.b_size, self.c_out, self.out_h(), self.out_w()]
    }
}

/// Ratio in `k_size <= DEPTHWISE_TAP_RATIO * groups`, the dispatch guard below.
///
/// With `groups == c_in == c_out` both lowerings move the same order of data
/// (`k_size * c * l_out` elements), so what actually differs is the number of backend
/// dispatches: `k_size` taps for the fast path against `groups` convolutions for the
/// reference loop. The predicate is therefore `k_size * cost_tap <= groups * cost_conv`,
/// i.e. `k_size <= ratio * groups` with `ratio = cost_conv / cost_tap`.
///
/// Measured on an M1 Max, depthwise conv1d, fast path vs reference loop
/// (`>1` means the fast path wins):
///
/// | `k_size` vs `groups` | cpu   | metal |
/// |----------------------|------:|------:|
/// | 1x                   | 3.97x | 1.18x |
/// | 2x                   | 1.77x | 2.38x |
/// | **4x**               | 1.06x | 1.25x |
/// | 16x                  | 1.07x | 0.32x |
/// | 32x                  | 0.58x | 0.21x |
///
/// 4x is the largest ratio that still wins on both backends, so that is the cutoff.
const DEPTHWISE_TAP_RATIO: usize = 4;

/// Whether the depthwise fast path can handle this 1d convolution.
///
/// `params.c_in` / `params.c_out` are already per-group, so both being 1 is exactly
/// `groups == c_in == c_out`, the depthwise case. Anything this returns `false` for keeps
/// using the historical `groups`-convolutions-plus-`cat` lowering, so a `false` can never
/// change a result -- only forgo a speedup.
fn depthwise_1d_applicable(params: &ParamsConv1D, groups: usize) -> bool {
    params.c_in == 1
        && params.c_out == 1
        && params.k_size >= 1
        && params.dilation >= 1
        // `stride > 1` would need a gather per tap. That measures 0.35-0.45x against the
        // reference loop on cpu (it wins on metal, but a device-dependent predicate is
        // worse than simply leaving strided convolutions on the unchanged path).
        && params.stride == 1
        && params.k_size <= DEPTHWISE_TAP_RATIO * groups
        // Guard l_out() against underflow; let the reference path report the error.
        && params.l_in + 2 * params.padding >= params.dilation * (params.k_size - 1) + 1
}

impl Tensor {
    fn conv1d_single_group(&self, kernel: &Self, params: &ParamsConv1D) -> Result<Self> {
        let storage =
            self.storage()
                .conv1d(self.layout(), &kernel.storage(), kernel.layout(), params)?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::Conv1D {
            arg,
            kernel,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 1D convolution over the input tensor.
    pub fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        self.conv1d_with_algo(kernel, padding, stride, dilation, groups, None)
    }

    /// Applies a 1D convolution over the input tensor.
    pub fn conv1d_with_algo(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        cudnn_fwd_algo: Option<CudnnFwdAlgo>,
    ) -> Result<Self> {
        let (c_out, c_in_k, k_size) = kernel.dims3()?;
        let (b_size, c_in, l_in) = self.dims3()?;
        if c_in != c_in_k * groups {
            Err(Error::Conv1dInvalidArgs {
                inp_shape: self.shape().clone(),
                k_shape: kernel.shape().clone(),
                padding,
                stride,
                msg: "the number of in-channels on the input doesn't match the kernel size",
            }
            .bt())?
        }

        let params = ParamsConv1D {
            b_size,
            l_in,
            c_out: c_out / groups,
            c_in: c_in / groups,
            k_size,
            padding,
            stride,
            dilation,
            cudnn_fwd_algo,
        };
        if groups == 1 {
            self.conv1d_single_group(kernel, &params)
        } else if depthwise_1d_applicable(&params, groups) {
            self.conv1d_depthwise(kernel, &params)
        } else {
            self.conv1d_grouped_loop(kernel, &params, groups)
        }
    }

    /// Reference lowering for a grouped 1d convolution: `groups` independent single-group
    /// convolutions, concatenated back together along the channel axis.
    ///
    /// This costs `O(groups)` backend dispatches, which is why depthwise convolutions
    /// (`groups == c_in == c_out`) take [`Tensor::conv1d_depthwise`] instead.
    fn conv1d_grouped_loop(
        &self,
        kernel: &Self,
        params: &ParamsConv1D,
        groups: usize,
    ) -> Result<Self> {
        let blocks = self.chunk(groups, 1)?;
        let kernel = kernel.chunk(groups, 0)?;
        let blocks = blocks
            .iter()
            .zip(&kernel)
            .map(|(block, kernel)| block.conv1d_single_group(kernel, params))
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&blocks, 1)
    }

    /// Depthwise 1d convolution (`groups == c_in == c_out`, i.e. one input and one output
    /// channel per group) expressed with `O(k_size)` tensor operations instead of `O(groups)`
    /// convolutions.
    ///
    /// For every kernel tap `j` the (optionally zero padded) input is sliced at offset
    /// `j * dilation`, scaled by that tap's per-channel weight broadcast over the spatial
    /// axis, and accumulated. Only the summation order differs from the reference lowering.
    ///
    /// Requires `stride == 1`; strided depthwise convolutions stay on the reference path.
    ///
    /// The caller must have checked [`depthwise_1d_applicable`].
    fn conv1d_depthwise(&self, kernel: &Self, params: &ParamsConv1D) -> Result<Self> {
        let (_b_size, c, _l_in) = self.dims3()?;
        let k_size = params.k_size;
        let l_out = params.l_out();
        // (c, 1, k_size) -> (k_size, c), so that tap j is a contiguous row.
        let taps = kernel.reshape((c, k_size))?.t()?.contiguous()?;
        let x = if params.padding > 0 {
            self.pad_with_zeros(2, params.padding, params.padding)?
        } else {
            self.clone()
        };
        let mut acc: Option<Tensor> = None;
        for j in 0..k_size {
            // stride == 1 is guaranteed by depthwise_1d_applicable, so tap j is exactly the
            // window starting at j * dilation -- a view, no copy.
            let slice = x.narrow(2, j * params.dilation, l_out)?;
            let tap = taps.narrow(0, j, 1)?.reshape((1, c, 1))?;
            let term = slice.broadcast_mul(&tap)?;
            acc = Some(match acc {
                None => term,
                Some(acc) => (acc + term)?,
            });
        }
        match acc {
            Some(acc) => Ok(acc),
            // k_size == 0 is rejected by depthwise_1d_applicable.
            None => crate::bail!("conv1d_depthwise: empty kernel"),
        }
    }

    fn conv_transpose1d_single_group(
        &self,
        kernel: &Self,
        params: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        let storage = self.storage().conv_transpose1d(
            self.layout(),
            &kernel.storage(),
            kernel.layout(),
            params,
        )?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::ConvTranspose1D {
            arg,
            kernel,
            padding: params.padding,
            output_padding: params.output_padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 1D transposed convolution over the input tensor.
    pub fn conv_transpose1d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let (c_in_k, c_out, k_size) = kernel.dims3()?;
        let (b_size, c_in, l_in) = self.dims3()?;
        if c_in != c_in_k {
            crate::bail!("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
        }
        if c_in % groups != 0 {
            crate::bail!("in_channel {c_in} is not divisible by the number of groups")
        }
        let params = ParamsConvTranspose1D {
            b_size,
            l_in,
            k_size,
            c_out,
            c_in: c_in / groups,
            padding,
            output_padding,
            stride,
            dilation,
        };
        if groups == 1 {
            self.conv_transpose1d_single_group(kernel, &params)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv_transpose1d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    fn conv2d_single_group(&self, kernel: &Self, params: &ParamsConv2D) -> Result<Self> {
        let storage =
            self.storage()
                .conv2d(self.layout(), &kernel.storage(), kernel.layout(), params)?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::Conv2D {
            arg,
            kernel,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 2D convolution over the input tensor.
    pub fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        self.conv2d_with_algo(kernel, padding, stride, dilation, groups, None)
    }

    pub fn conv2d_with_algo(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        cudnn_fwd_algo: Option<CudnnFwdAlgo>,
    ) -> Result<Self> {
        let (b_size, c_in, i_h, i_w) = self.dims4()?;
        let (c_out, c_in_k, k_h, k_w) = kernel.dims4()?;
        if c_in != c_in_k * groups {
            crate::bail!(
                "in_channel mismatch between input ({c_in}, groups {groups}) and kernel ({c_in_k})"
            )
        }
        let params = ParamsConv2D {
            b_size,
            i_h,
            i_w,
            k_h,
            k_w,
            c_out: c_out / groups,
            c_in: c_in / groups,
            padding,
            stride,
            dilation,
            cudnn_fwd_algo,
        };
        if groups == 1 {
            self.conv2d_single_group(kernel, &params)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv2d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    /// Applies a 2D transposed convolution over the input tensor.
    pub fn conv_transpose2d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
    ) -> Result<Self> {
        let (b_size, c_in, i_h, i_w) = self.dims4()?;
        let (c_in_k, c_out, k_h, k_w) = kernel.dims4()?;
        if c_in != c_in_k {
            crate::bail!("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
        }
        let params = ParamsConvTranspose2D {
            b_size,
            i_h,
            i_w,
            k_h,
            k_w,
            c_out,
            c_in,
            padding,
            output_padding,
            stride,
            dilation,
        };
        let storage = self.storage().conv_transpose2d(
            self.layout(),
            &kernel.storage(),
            kernel.layout(),
            &params,
        )?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::ConvTranspose2D {
            arg,
            kernel,
            padding: params.padding,
            output_padding: params.output_padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }
}

#[cfg(test)]
mod depthwise_tests {
    use super::*;
    use crate::{test_device, DType, Device, IndexOp};

    // Every test below is instantiated per backend by `test_device!`, which generates a
    // separate `_cpu` / `_cuda` / `_metal` test and propagates `Device::new_metal(0)?`.
    // A machine where the accelerator cannot be opened therefore FAILS the metal tests
    // rather than silently reporting green having exercised cpu only.

    /// Deterministic test data: candle's `Tensor::randn` cannot be seeded on the cpu
    /// backend, and a flaky tolerance is worse than no test.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        }

        fn tensor<S: Into<crate::Shape>>(&mut self, shape: S, dev: &Device) -> Result<Tensor> {
            let shape: crate::Shape = shape.into();
            let v: Vec<f32> = (0..shape.elem_count()).map(|_| self.next_f32()).collect();
            Tensor::from_vec(v, shape, dev)
        }
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        assert_eq!(a.dims(), b.dims());
        let d = (a - b)?.abs()?.flatten_all()?.max(0)?;
        d.to_scalar::<f32>()
    }

    /// Brute force depthwise conv1d evaluated in f64 on the host. Independent of every
    /// candle convolution code path, so it is the arbiter when the two lowerings disagree.
    fn brute_force_1d(x: &Tensor, w: &Tensor, params: &ParamsConv1D) -> Result<Tensor> {
        let dev = x.device().clone();
        // f64 is not available on every backend, so evaluate on the host.
        let xv = x
            .to_device(&Device::Cpu)?
            .contiguous()?
            .to_dtype(DType::F64)?
            .to_vec3::<f64>()?;
        let wv = w
            .to_device(&Device::Cpu)?
            .contiguous()?
            .to_dtype(DType::F64)?
            .to_vec3::<f64>()?;
        let (b, c, l_in) = (xv.len(), xv[0].len(), xv[0][0].len());
        let l_out = params.l_out();
        let mut out = vec![0f64; b * c * l_out];
        for bi in 0..b {
            for ci in 0..c {
                for o in 0..l_out {
                    let mut acc = 0f64;
                    for j in 0..params.k_size {
                        let pos = (o * params.stride + j * params.dilation) as i64
                            - params.padding as i64;
                        if pos >= 0 && (pos as usize) < l_in {
                            acc += xv[bi][ci][pos as usize] * wv[ci][0][j];
                        }
                    }
                    out[(bi * c + ci) * l_out + o] = acc;
                }
            }
        }
        Tensor::from_vec(out, (b, c, l_out), &Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_device(&dev)
    }

    /// Brute force depthwise conv2d evaluated in f64 on the host.
    fn brute_force_2d(x: &Tensor, w: &Tensor, params: &ParamsConv2D) -> Result<Tensor> {
        let dev = x.device().clone();
        let xv = x
            .to_device(&Device::Cpu)?
            .contiguous()?
            .to_dtype(DType::F64)?
            .flatten_all()?
            .to_vec1::<f64>()?;
        let wv = w
            .to_device(&Device::Cpu)?
            .contiguous()?
            .to_dtype(DType::F64)?
            .flatten_all()?
            .to_vec1::<f64>()?;
        let (b, c, i_h, i_w) = (params.b_size, x.dims()[1], params.i_h, params.i_w);
        let (k_h, k_w) = (params.k_h, params.k_w);
        let (o_h, o_w) = (params.out_h(), params.out_w());
        let mut out = vec![0f64; b * c * o_h * o_w];
        for bi in 0..b {
            for ci in 0..c {
                for oh in 0..o_h {
                    for ow in 0..o_w {
                        let mut acc = 0f64;
                        for jh in 0..k_h {
                            for jw in 0..k_w {
                                let ph = (oh * params.stride + jh * params.dilation) as i64
                                    - params.padding as i64;
                                let pw = (ow * params.stride + jw * params.dilation) as i64
                                    - params.padding as i64;
                                if ph >= 0 && (ph as usize) < i_h && pw >= 0 && (pw as usize) < i_w
                                {
                                    let xi =
                                        ((bi * c + ci) * i_h + ph as usize) * i_w + pw as usize;
                                    let wi = (ci * k_h + jh) * k_w + jw;
                                    acc += xv[xi] * wv[wi];
                                }
                            }
                        }
                        out[((bi * c + ci) * o_h + oh) * o_w + ow] = acc;
                    }
                }
            }
        }
        Tensor::from_vec(out, (b, c, o_h, o_w), &Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_device(&dev)
    }

    /// 1d cases the fast path must handle: (name, b, c, l_in, k, padding, dilation).
    /// `stride` is always 1 -- strided depthwise convolutions fall back by design, and are
    /// covered by `depthwise_dispatch_boundary`.
    const CASES_1D: &[(&str, usize, usize, usize, usize, usize, usize)] = &[
        // Qwen3.5 GatedDeltaNet, the shape that motivated this fast path.
        ("model-decode", 1, 6144, 4, 4, 0, 1),
        ("model-prefill", 1, 6144, 29, 4, 0, 1),
        ("small-odd", 2, 5, 7, 3, 0, 1),
        ("k1", 1, 3, 6, 1, 0, 1),
        ("pad1", 1, 3, 9, 2, 1, 1),
        ("pad2", 2, 4, 11, 3, 2, 1),
        ("dilation2", 1, 6, 13, 3, 0, 2),
        ("pad-dilation", 2, 7, 15, 4, 2, 3),
        ("big-dilation", 1, 8, 20, 3, 0, 5),
        ("l_out-1", 1, 4, 4, 4, 0, 1),
    ];

    fn params_1d(
        b_size: usize,
        l_in: usize,
        k: usize,
        padding: usize,
        stride: usize,
        dilation: usize,
    ) -> ParamsConv1D {
        ParamsConv1D {
            b_size,
            l_in,
            c_out: 1,
            c_in: 1,
            k_size: k,
            padding,
            stride,
            dilation,
            cudnn_fwd_algo: None,
        }
    }

    fn depthwise_conv1d_matches_grouped_loop(dev: &Device) -> Result<()> {
        let mut rng = Lcg::new(11);
        for &(name, b, c, l_in, k, padding, dilation) in CASES_1D {
            let x = rng.tensor((b, c, l_in), dev)?;
            let w = rng.tensor((c, 1, k), dev)?;
            let params = params_1d(b, l_in, k, padding, 1, dilation);
            assert!(
                depthwise_1d_applicable(&params, c),
                "{name}: expected the fast path to be applicable"
            );
            let want = x.conv1d_grouped_loop(&w, &params, c)?;
            let got = x.conv1d_depthwise(&w, &params)?;
            assert_eq!(got.dims(), want.dims(), "{name}: shape");
            let diff = max_abs_diff(&got, &want)?;
            assert!(diff <= 1e-5, "{name}: max abs diff {diff}");
            // And the public entry point must agree too.
            let public = x.conv1d(&w, padding, 1, dilation, c)?;
            let diff = max_abs_diff(&public, &want)?;
            assert!(diff <= 1e-5, "{name}: public max abs diff {diff}");
            // ... and both must match an independent host side f64 evaluation.
            let brute = brute_force_1d(&x, &w, &params)?;
            let diff = max_abs_diff(&got, &brute)?;
            assert!(diff <= 1e-4, "{name}: brute force max abs diff {diff}");
        }
        Ok(())
    }

    /// Independent reference: convolve each channel on its own with the public API.
    fn depthwise_conv1d_matches_per_channel_reference(dev: &Device) -> Result<()> {
        let mut rng = Lcg::new(22);
        for &(name, b, c, l_in, k, padding, dilation) in CASES_1D {
            if c > 64 {
                continue; // too slow as a per-channel reference
            }
            let x = rng.tensor((b, c, l_in), dev)?;
            let w = rng.tensor((c, 1, k), dev)?;
            let mut chans = Vec::with_capacity(c);
            for i in 0..c {
                let xi = x.narrow(1, i, 1)?;
                let wi = w.narrow(0, i, 1)?;
                chans.push(xi.conv1d(&wi, padding, 1, dilation, 1)?);
            }
            let want = Tensor::cat(&chans, 1)?;
            let got = x.conv1d(&w, padding, 1, dilation, c)?;
            assert_eq!(got.dims(), want.dims(), "{name}: shape");
            let diff = max_abs_diff(&got, &want)?;
            assert!(diff <= 1e-5, "{name}: max abs diff {diff}");
        }
        Ok(())
    }

    /// A hand-computed depthwise convolution, so that the suite does not only compare two
    /// implementations against each other.
    fn depthwise_conv1d_hand_computed(dev: &Device) -> Result<()> {
        // c = 2, l_in = 5, k = 2, padding = 1, stride = 1, dilation = 1 -> l_out = 6
        let x = Tensor::new(&[[[1f32, 2., 3., 4., 5.], [10., 20., 30., 40., 50.]]], dev)?;
        let w = Tensor::new(&[[[1f32, 2.]], [[3f32, -1.]]], dev)?;
        let got = x.conv1d(&w, 1, 1, 1, 2)?;
        // padded ch0: 0 1 2 3 4 5 0, taps [1, 2] -> out[o] = pad[o] + 2*pad[o+1]
        // padded ch1: 0 10 .. 50 0,  taps [3,-1] -> out[o] = 3*pad[o] - pad[o+1]
        let want = Tensor::new(
            &[[
                [2f32, 5., 8., 11., 14., 5.],
                [-10f32, 10., 30., 50., 70., 150.],
            ]],
            dev,
        )?;
        let diff = max_abs_diff(&got, &want)?;
        assert!(diff <= 1e-5, "max abs diff {diff}");
        Ok(())
    }

    fn depthwise_conv1d_non_contiguous_inputs(dev: &Device) -> Result<()> {
        let mut rng = Lcg::new(33);
        let c = 6;
        // Non contiguous input: build (b, l, c) then transpose.
        let x = rng.tensor((2, 11, c), dev)?.transpose(1, 2)?;
        assert!(!x.is_contiguous());
        // Non contiguous kernel: (k, 1, c) transposed to (c, 1, k).
        let w = rng.tensor((3, 1, c), dev)?.transpose(0, 2)?;
        assert!(!w.is_contiguous());
        let params = params_1d(2, 11, 3, 1, 1, 1);
        // Checked against the host side f64 evaluation rather than against
        // conv1d_grouped_loop: the backend conv kernels mishandle a non contiguous
        // *kernel* tensor (see the pre-existing upstream failure
        // conv2d_grad_noncontiguous_kernel), so the reference lowering is not
        // trustworthy here. The fast path is, and this pins that down.
        let brute = brute_force_1d(&x, &w, &params)?;
        let got = x.conv1d_depthwise(&w, &params)?;
        let diff = max_abs_diff(&got, &brute)?;
        assert!(diff <= 1e-4, "max abs diff {diff}");
        // With a contiguous kernel the two lowerings must agree exactly as usual.
        let wc = w.contiguous()?;
        let want = x.conv1d_grouped_loop(&wc, &params, c)?;
        let got = x.conv1d_depthwise(&wc, &params)?;
        let diff = max_abs_diff(&got, &want)?;
        assert!(diff <= 1e-5, "contiguous-kernel max abs diff {diff}");
        Ok(())
    }

    fn depthwise_conv1d_dtypes(dev: &Device) -> Result<()> {
        let mut rng = Lcg::new(44);
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::F64] {
            if !dev.is_cpu() && dtype == DType::F64 {
                continue;
            }
            let c = 16;
            let x = rng.tensor((1, c, 9), dev)?.to_dtype(dtype)?;
            let w = rng.tensor((c, 1, 3), dev)?.to_dtype(dtype)?;
            let params = params_1d(1, 9, 3, 0, 1, 1);
            // The reference lowering goes through matmul, which candle does not
            // implement for bf16 on cpu, so compare against the f64 host evaluation.
            let want = brute_force_1d(&x, &w, &params)?;
            let got = x.conv1d_depthwise(&w, &params)?.to_dtype(DType::F32)?;
            let diff = max_abs_diff(&got, &want)?;
            // Tolerance scaled by the output magnitude and the dtype's mantissa.
            let scale = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let rel = match dtype {
                DType::BF16 => 3e-2,
                DType::F16 => 3e-3,
                _ => 1e-6,
            };
            let tol = rel * scale.max(1.0);
            assert!(diff <= tol, "{dtype:?}: max abs diff {diff} > {tol}");
        }
        Ok(())
    }

    /// The exact boundary of the fast path.
    ///
    /// These assertions encode a *measured* policy, not a guess -- see
    /// [`DEPTHWISE_TAP_RATIO`] for the table. Everything the predicate rejects keeps using
    /// the unchanged reference lowering, and is checked here to still be numerically right.
    fn depthwise_dispatch_boundary(dev: &Device) -> Result<()> {
        // Depthwise, stride 1: the fast path, whatever the padding and dilation.
        assert!(depthwise_1d_applicable(&params_1d(1, 20, 4, 0, 1, 1), 4));
        assert!(depthwise_1d_applicable(&params_1d(1, 20, 3, 2, 1, 3), 8));
        assert!(depthwise_1d_applicable(&params_1d(1, 8, 4, 0, 1, 1), 6144));

        // k_size == 4 * groups is the last measured win on both cpu and metal.
        assert!(depthwise_1d_applicable(&params_1d(1, 40, 16, 0, 1, 1), 4));
        // One tap further and the reference loop is cheaper.
        assert!(!depthwise_1d_applicable(&params_1d(1, 40, 17, 0, 1, 1), 4));

        // stride > 1 needs a gather, which regresses on cpu (0.35-0.45x measured).
        assert!(!depthwise_1d_applicable(&params_1d(1, 20, 3, 0, 2, 1), 8));
        assert!(!depthwise_1d_applicable(&params_1d(1, 20, 3, 1, 3, 1), 8));

        // Degenerate: input too short for the dilated kernel. Must not be admitted, so
        // that l_out() is never evaluated on an underflowing subtraction.
        assert!(!depthwise_1d_applicable(&params_1d(1, 3, 4, 0, 1, 4), 8));

        // Not depthwise: two channels per group.
        let not_depthwise = ParamsConv1D {
            b_size: 1,
            l_in: 10,
            c_out: 2,
            c_in: 2,
            k_size: 3,
            padding: 0,
            stride: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        assert!(!depthwise_1d_applicable(&not_depthwise, 4));

        // Every fall-back configuration must still come out of the public API correct.
        let mut rng = Lcg::new(55);
        let x = rng.tensor((1, 8, 10), dev)?;
        let w = rng.tensor((8, 2, 3), dev)?;
        let want = x.conv1d_grouped_loop(&w, &not_depthwise, 4)?;
        let got = x.conv1d(&w, 0, 1, 1, 4)?;
        assert!(
            max_abs_diff(&got, &want)? <= 1e-5,
            "not-depthwise fall back"
        );

        // stride 2 falls back, and is still numerically right.
        let params = params_1d(1, 21, 3, 1, 2, 1);
        let x = rng.tensor((1, 8, 21), dev)?;
        let w = rng.tensor((8, 1, 3), dev)?;
        let brute = brute_force_1d(&x, &w, &params)?;
        let got = x.conv1d(&w, 1, 2, 1, 8)?;
        assert!(
            max_abs_diff(&got, &brute)? <= 1e-4,
            "stride-2 fall back must still be correct"
        );

        // k_size far above the ratio falls back, and is still numerically right.
        let params = params_1d(1, 40, 17, 0, 1, 1);
        let x = rng.tensor((1, 4, 40), dev)?;
        let w = rng.tensor((4, 1, 17), dev)?;
        let brute = brute_force_1d(&x, &w, &params)?;
        let got = x.conv1d(&w, 0, 1, 1, 4)?;
        assert!(
            max_abs_diff(&got, &brute)? <= 1e-4,
            "large-kernel fall back must still be correct"
        );
        Ok(())
    }

    /// Grouped conv2d has no fast path -- measurement showed the tap formulation loses at
    /// the shapes that matter. This pins the unchanged lowering so the removal stays
    /// removed and the 2d loop keeps producing the right answer.
    fn depthwise_conv2d_uses_unchanged_lowering(dev: &Device) -> Result<()> {
        let mut rng = Lcg::new(77);
        for &(name, b, c, i_h, i_w, k_h, k_w, padding, stride, dilation) in &[
            (
                "mobilenet-3x3",
                2usize,
                32usize,
                14usize,
                14usize,
                3usize,
                3usize,
                1usize,
                1usize,
                1usize,
            ),
            ("mobilenet-3x3-s2", 1, 16, 15, 15, 3, 3, 1, 2, 1),
            ("small-odd", 1, 8, 7, 6, 2, 3, 0, 1, 1),
            ("dilation", 1, 9, 12, 11, 3, 3, 2, 1, 2),
        ] {
            let x = rng.tensor((b, c, i_h, i_w), dev)?;
            let k = rng.tensor((c, 1, k_h, k_w), dev)?;
            let params = ParamsConv2D {
                b_size: b,
                i_h,
                i_w,
                k_h,
                k_w,
                c_out: 1,
                c_in: 1,
                padding,
                stride,
                dilation,
                cudnn_fwd_algo: None,
            };
            let got = x.conv2d(&k, padding, stride, dilation, c)?;
            let brute = brute_force_2d(&x, &k, &params)?;
            assert_eq!(got.dims(), brute.dims(), "{name}: shape");
            let diff = max_abs_diff(&got, &brute)?;
            assert!(diff <= 1e-4, "{name}: max abs diff {diff}");
        }
        Ok(())
    }

    fn depthwise_conv1d_backprop(dev: &Device) -> Result<()> {
        let mut rng = Lcg::new(66);
        let c = 6;
        let x = crate::Var::from_tensor(&rng.tensor((2, c, 9), dev)?)?;
        let w = crate::Var::from_tensor(&rng.tensor((c, 1, 3), dev)?)?;
        let y = x.as_tensor().conv1d(w.as_tensor(), 1, 1, 1, c)?;
        let loss = (&y * &y)?.sum_all()?;
        let grads = loss.backward()?;
        let gx = grads.get(&x).expect("no grad for x").clone();
        let gw = grads.get(&w).expect("no grad for w").clone();
        assert_eq!(gx.dims(), x.dims());
        assert_eq!(gw.dims(), w.dims());
        // Finite differences on a couple of entries of w. Note this cross-checks autograd
        // against the *same* forward function, so it verifies that gradients flow through
        // the new op graph -- it is not an independent forward-correctness check.
        let eps = 1e-2f32;
        let base = loss.to_scalar::<f32>()?;
        for i in [0usize, 3] {
            let mut wv = w.flatten_all()?.to_vec1::<f32>()?;
            wv[i] += eps;
            let wp = Tensor::from_vec(wv, (c, 1, 3), dev)?;
            let yp = x.as_tensor().conv1d(&wp, 1, 1, 1, c)?;
            let lp = (&yp * &yp)?.sum_all()?.to_scalar::<f32>()?;
            let num = (lp - base) / eps;
            let ana = gw.flatten_all()?.i(i)?.to_scalar::<f32>()?;
            assert!(
                (num - ana).abs() <= 0.15 * (1.0 + ana.abs()),
                "grad w[{i}]: numeric {num} analytic {ana}"
            );
        }
        Ok(())
    }

    test_device!(
        depthwise_conv1d_matches_grouped_loop,
        depthwise_conv1d_matches_grouped_loop_cpu,
        depthwise_conv1d_matches_grouped_loop_gpu,
        depthwise_conv1d_matches_grouped_loop_metal
    );
    test_device!(
        depthwise_conv1d_matches_per_channel_reference,
        depthwise_conv1d_matches_per_channel_reference_cpu,
        depthwise_conv1d_matches_per_channel_reference_gpu,
        depthwise_conv1d_matches_per_channel_reference_metal
    );
    test_device!(
        depthwise_conv1d_hand_computed,
        depthwise_conv1d_hand_computed_cpu,
        depthwise_conv1d_hand_computed_gpu,
        depthwise_conv1d_hand_computed_metal
    );
    test_device!(
        depthwise_conv1d_non_contiguous_inputs,
        depthwise_conv1d_non_contiguous_inputs_cpu,
        depthwise_conv1d_non_contiguous_inputs_gpu,
        depthwise_conv1d_non_contiguous_inputs_metal
    );
    test_device!(
        depthwise_conv1d_dtypes,
        depthwise_conv1d_dtypes_cpu,
        depthwise_conv1d_dtypes_gpu,
        depthwise_conv1d_dtypes_metal
    );
    test_device!(
        depthwise_dispatch_boundary,
        depthwise_dispatch_boundary_cpu,
        depthwise_dispatch_boundary_gpu,
        depthwise_dispatch_boundary_metal
    );
    test_device!(
        depthwise_conv2d_uses_unchanged_lowering,
        depthwise_conv2d_uses_unchanged_lowering_cpu,
        depthwise_conv2d_uses_unchanged_lowering_gpu,
        depthwise_conv2d_uses_unchanged_lowering_metal
    );
    test_device!(
        depthwise_conv1d_backprop,
        depthwise_conv1d_backprop_cpu,
        depthwise_conv1d_backprop_gpu,
        depthwise_conv1d_backprop_metal
    );
}
