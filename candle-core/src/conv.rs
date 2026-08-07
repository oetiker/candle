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

/// Whether the depthwise fast path can handle this 1d convolution.
///
/// `params.c_in` / `params.c_out` are already per-group, so both being 1 is exactly
/// `groups == c_in == c_out`, the depthwise case. Anything this returns `false` for keeps
/// using the historical `groups`-convolutions-plus-`cat` lowering.
fn depthwise_1d_applicable(params: &ParamsConv1D, groups: usize) -> bool {
    params.c_in == 1
        && params.c_out == 1
        && params.k_size >= 1
        && params.stride >= 1
        && params.dilation >= 1
        // The fast path costs O(k_size) ops; below this the reference loop is cheaper.
        && params.k_size <= groups
        // Guard l_out() against underflow; let the reference path report the error.
        && params.l_in + 2 * params.padding >= params.dilation * (params.k_size - 1) + 1
}

/// Whether the depthwise fast path can handle this 2d convolution, see
/// [`depthwise_1d_applicable`].
fn depthwise_2d_applicable(params: &ParamsConv2D, groups: usize) -> bool {
    params.c_in == 1
        && params.c_out == 1
        && params.k_h >= 1
        && params.k_w >= 1
        && params.stride >= 1
        && params.dilation >= 1
        && params.k_h * params.k_w <= groups
        && params.i_h + 2 * params.padding >= params.dilation * (params.k_h - 1) + 1
        && params.i_w + 2 * params.padding >= params.dilation * (params.k_w - 1) + 1
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
        // For stride > 1 the taps are gathered rather than sliced; the gather indices are the
        // same for every tap once the tap offset has been narrowed away.
        let ids = if params.stride == 1 {
            None
        } else {
            let ids: Vec<u32> = (0..l_out).map(|i| (i * params.stride) as u32).collect();
            Some(Tensor::from_vec(ids, l_out, self.device())?)
        };
        let mut acc: Option<Tensor> = None;
        for j in 0..k_size {
            let offset = j * params.dilation;
            let slice = match ids.as_ref() {
                None => x.narrow(2, offset, l_out)?,
                Some(ids) => {
                    let span = (l_out - 1) * params.stride + 1;
                    // index_select requires a contiguous source on every backend.
                    x.narrow(2, offset, span)?
                        .contiguous()?
                        .index_select(ids, 2)?
                }
            };
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
        } else if depthwise_2d_applicable(&params, groups) {
            self.conv2d_depthwise(kernel, &params)
        } else {
            self.conv2d_grouped_loop(kernel, &params, groups)
        }
    }

    /// Reference lowering for a grouped 2d convolution, see [`Tensor::conv1d_grouped_loop`].
    fn conv2d_grouped_loop(
        &self,
        kernel: &Self,
        params: &ParamsConv2D,
        groups: usize,
    ) -> Result<Self> {
        let blocks = self.chunk(groups, 1)?;
        let kernel = kernel.chunk(groups, 0)?;
        let blocks = blocks
            .iter()
            .zip(&kernel)
            .map(|(block, kernel)| block.conv2d_single_group(kernel, params))
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&blocks, 1)
    }

    /// Depthwise 2d convolution, the 2d twin of [`Tensor::conv1d_depthwise`]: `O(k_h * k_w)`
    /// tensor operations instead of `O(groups)` convolutions.
    ///
    /// The caller must have checked [`depthwise_2d_applicable`].
    fn conv2d_depthwise(&self, kernel: &Self, params: &ParamsConv2D) -> Result<Self> {
        let (_b_size, c, _i_h, _i_w) = self.dims4()?;
        let (k_h, k_w) = (params.k_h, params.k_w);
        let (out_h, out_w) = (params.out_h(), params.out_w());
        // (c, 1, k_h, k_w) -> (k_h * k_w, c), so that tap (jh, jw) is a contiguous row.
        let taps = kernel.reshape((c, k_h * k_w))?.t()?.contiguous()?;
        let x = if params.padding > 0 {
            self.pad_with_zeros(2, params.padding, params.padding)?
                .pad_with_zeros(3, params.padding, params.padding)?
        } else {
            self.clone()
        };
        let ids = if params.stride == 1 {
            None
        } else {
            let mk = |n: usize| -> Result<Tensor> {
                let ids: Vec<u32> = (0..n).map(|i| (i * params.stride) as u32).collect();
                Tensor::from_vec(ids, n, self.device())
            };
            Some((mk(out_h)?, mk(out_w)?))
        };
        let mut acc: Option<Tensor> = None;
        for jh in 0..k_h {
            for jw in 0..k_w {
                let (off_h, off_w) = (jh * params.dilation, jw * params.dilation);
                let slice = match ids.as_ref() {
                    None => x.narrow(2, off_h, out_h)?.narrow(3, off_w, out_w)?,
                    Some((ids_h, ids_w)) => {
                        let span_h = (out_h - 1) * params.stride + 1;
                        let span_w = (out_w - 1) * params.stride + 1;
                        x.narrow(2, off_h, span_h)?
                            .narrow(3, off_w, span_w)?
                            .contiguous()?
                            .index_select(ids_h, 2)?
                            .contiguous()?
                            .index_select(ids_w, 3)?
                    }
                };
                let tap = taps.narrow(0, jh * k_w + jw, 1)?.reshape((1, c, 1, 1))?;
                let term = slice.broadcast_mul(&tap)?;
                acc = Some(match acc {
                    None => term,
                    Some(acc) => (acc + term)?,
                });
            }
        }
        match acc {
            Some(acc) => Ok(acc),
            None => crate::bail!("conv2d_depthwise: empty kernel"),
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
    use crate::{DType, Device, IndexOp};

    fn devices() -> Vec<(&'static str, Device)> {
        let mut devices = vec![("cpu", Device::Cpu)];
        #[cfg(feature = "metal")]
        if let Ok(d) = Device::new_metal(0) {
            devices.push(("metal", d));
        }
        #[cfg(feature = "cuda")]
        if let Ok(d) = Device::new_cuda(0) {
            devices.push(("cuda", d));
        }
        devices
    }

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

    /// 1d cases: (name, b, c, l_in, k, padding, stride, dilation)
    const CASES_1D: &[(&str, usize, usize, usize, usize, usize, usize, usize)] = &[
        // Qwen3.5 GatedDeltaNet, the shape that motivated this fast path.
        ("model-decode", 1, 6144, 4, 4, 0, 1, 1),
        ("model-prefill", 1, 6144, 29, 4, 0, 1, 1),
        ("small-odd", 2, 5, 7, 3, 0, 1, 1),
        ("k1", 1, 3, 6, 1, 0, 1, 1),
        ("pad1", 1, 3, 9, 2, 1, 1, 1),
        ("pad2", 2, 4, 11, 3, 2, 1, 1),
        ("dilation2", 1, 6, 13, 3, 0, 1, 2),
        ("pad-dilation", 2, 7, 15, 4, 2, 1, 3),
        ("stride2", 1, 8, 16, 3, 1, 2, 1),
        ("stride3-pad-dilation", 2, 5, 17, 3, 2, 3, 2),
        ("l_out-1", 1, 4, 4, 4, 0, 1, 1),
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

    #[test]
    fn depthwise_conv1d_matches_grouped_loop() -> Result<()> {
        for (dev_name, dev) in devices() {
            let mut rng = Lcg::new(11);
            for &(name, b, c, l_in, k, padding, stride, dilation) in CASES_1D {
                let x = rng.tensor((b, c, l_in), &dev)?;
                let w = rng.tensor((c, 1, k), &dev)?;
                let params = params_1d(b, l_in, k, padding, stride, dilation);
                assert!(
                    depthwise_1d_applicable(&params, c),
                    "{name}: expected the fast path to be applicable"
                );
                let want = x.conv1d_grouped_loop(&w, &params, c)?;
                let got = x.conv1d_depthwise(&w, &params)?;
                assert_eq!(got.dims(), want.dims(), "{dev_name}/{name}: shape");
                let diff = max_abs_diff(&got, &want)?;
                assert!(diff <= 1e-5, "{dev_name}/{name}: max abs diff {diff}");
                // And the public entry point must agree too.
                let public = x.conv1d(&w, padding, stride, dilation, c)?;
                let diff = max_abs_diff(&public, &want)?;
                assert!(
                    diff <= 1e-5,
                    "{dev_name}/{name}: public max abs diff {diff}"
                );
                // ... and both must match an independent host side f64 evaluation.
                let brute = brute_force_1d(&x, &w, &params)?;
                let diff = max_abs_diff(&got, &brute)?;
                assert!(
                    diff <= 1e-4,
                    "{dev_name}/{name}: brute force max abs diff {diff}"
                );
            }
        }
        Ok(())
    }

    /// Independent reference: convolve each channel on its own with the public API.
    #[test]
    fn depthwise_conv1d_matches_per_channel_reference() -> Result<()> {
        for (dev_name, dev) in devices() {
            let mut rng = Lcg::new(22);
            for &(name, b, c, l_in, k, padding, stride, dilation) in CASES_1D {
                if c > 64 {
                    continue; // too slow as a per-channel reference
                }
                let x = rng.tensor((b, c, l_in), &dev)?;
                let w = rng.tensor((c, 1, k), &dev)?;
                let mut chans = Vec::with_capacity(c);
                for i in 0..c {
                    let xi = x.narrow(1, i, 1)?;
                    let wi = w.narrow(0, i, 1)?;
                    chans.push(xi.conv1d(&wi, padding, stride, dilation, 1)?);
                }
                let want = Tensor::cat(&chans, 1)?;
                let got = x.conv1d(&w, padding, stride, dilation, c)?;
                assert_eq!(got.dims(), want.dims(), "{dev_name}/{name}: shape");
                let diff = max_abs_diff(&got, &want)?;
                assert!(diff <= 1e-5, "{dev_name}/{name}: max abs diff {diff}");
            }
        }
        Ok(())
    }

    /// A hand-computed depthwise convolution, so that the suite does not only compare two
    /// implementations against each other.
    #[test]
    fn depthwise_conv1d_hand_computed() -> Result<()> {
        for (dev_name, dev) in devices() {
            // c = 2, l_in = 5, k = 2, padding = 1, stride = 2, dilation = 1 -> l_out = 3
            let x = Tensor::new(&[[[1f32, 2., 3., 4., 5.], [10., 20., 30., 40., 50.]]], &dev)?;
            let w = Tensor::new(&[[[1f32, 2.]], [[3f32, -1.]]], &dev)?;
            let got = x.conv1d(&w, 1, 2, 1, 2)?;
            // padded ch0: 0 1 2 3 4 5 0 ; taps at 0,2,4 -> [0*1+1*2, 2*1+3*2, 4*1+5*2]
            // padded ch1: 0 10 .. 50 0 ; -> [0*3-10, 20*3-30, 40*3-50]
            let want = Tensor::new(&[[[2f32, 8., 14.], [-10f32, 30., 70.]]], &dev)?;
            let diff = max_abs_diff(&got, &want)?;
            assert!(diff <= 1e-5, "{dev_name}: max abs diff {diff}");
        }
        Ok(())
    }

    #[test]
    fn depthwise_conv1d_non_contiguous_inputs() -> Result<()> {
        for (dev_name, dev) in devices() {
            let mut rng = Lcg::new(33);
            let c = 6;
            // Non contiguous input: build (b, l, c) then transpose.
            let x = rng.tensor((2, 11, c), &dev)?.transpose(1, 2)?;
            assert!(!x.is_contiguous());
            // Non contiguous kernel: (k, 1, c) transposed to (c, 1, k).
            let w = rng.tensor((3, 1, c), &dev)?.transpose(0, 2)?;
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
            assert!(diff <= 1e-4, "{dev_name}: max abs diff {diff}");
            // With a contiguous kernel the two lowerings must agree exactly as usual.
            let wc = w.contiguous()?;
            let want = x.conv1d_grouped_loop(&wc, &params, c)?;
            let got = x.conv1d_depthwise(&wc, &params)?;
            let diff = max_abs_diff(&got, &want)?;
            assert!(
                diff <= 1e-5,
                "{dev_name}: contiguous-kernel max abs diff {diff}"
            );
        }
        Ok(())
    }

    #[test]
    fn depthwise_conv1d_dtypes() -> Result<()> {
        for (dev_name, dev) in devices() {
            let mut rng = Lcg::new(44);
            for dtype in [DType::F32, DType::F16, DType::BF16, DType::F64] {
                if dev.is_metal() && dtype == DType::F64 {
                    continue;
                }
                let c = 16;
                let x = rng.tensor((1, c, 9), &dev)?.to_dtype(dtype)?;
                let w = rng.tensor((c, 1, 3), &dev)?.to_dtype(dtype)?;
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
                assert!(
                    diff <= tol,
                    "{dev_name}/{dtype:?}: max abs diff {diff} > {tol}"
                );
            }
        }
        Ok(())
    }

    /// Configurations that must keep using the reference lowering.
    #[test]
    fn depthwise_fallbacks() -> Result<()> {
        let dev = Device::Cpu;
        let mut rng = Lcg::new(55);
        // Not depthwise: 2 channels per group.
        let x = rng.tensor((1, 8, 10), &dev)?;
        let w = rng.tensor((8, 2, 3), &dev)?;
        let params = ParamsConv1D {
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
        assert!(!depthwise_1d_applicable(&params, 4));
        let want = x.conv1d_grouped_loop(&w, &params, 4)?;
        let got = x.conv1d(&w, 0, 1, 1, 4)?;
        assert!(max_abs_diff(&got, &want)? <= 1e-5);

        // Depthwise but k_size > groups: the reference loop is cheaper there.
        let params = ParamsConv1D {
            b_size: 1,
            l_in: 20,
            c_out: 1,
            c_in: 1,
            k_size: 5,
            padding: 0,
            stride: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        assert!(!depthwise_1d_applicable(&params, 2));
        assert!(depthwise_1d_applicable(&params, 5));
        let x = rng.tensor((1, 2, 20), &dev)?;
        let w = rng.tensor((2, 1, 5), &dev)?;
        let want = x.conv1d_grouped_loop(&w, &params, 2)?;
        let got = x.conv1d(&w, 0, 1, 1, 2)?;
        assert!(max_abs_diff(&got, &want)? <= 1e-5);
        Ok(())
    }

    #[test]
    fn depthwise_conv1d_backprop() -> Result<()> {
        let dev = Device::Cpu;
        let mut rng = Lcg::new(66);
        let c = 6;
        let x = crate::Var::from_tensor(&rng.tensor((2, c, 9), &dev)?)?;
        let w = crate::Var::from_tensor(&rng.tensor((c, 1, 3), &dev)?)?;
        let y = x.as_tensor().conv1d(w.as_tensor(), 1, 1, 1, c)?;
        let loss = (&y * &y)?.sum_all()?;
        let grads = loss.backward()?;
        let gx = grads.get(&x).expect("no grad for x").clone();
        let gw = grads.get(&w).expect("no grad for w").clone();
        assert_eq!(gx.dims(), x.dims());
        assert_eq!(gw.dims(), w.dims());
        // Finite differences on a couple of entries of w.
        let eps = 1e-2f32;
        let base = loss.to_scalar::<f32>()?;
        for i in [0usize, 3] {
            let mut wv = w.flatten_all()?.to_vec1::<f32>()?;
            wv[i] += eps;
            let wp = Tensor::from_vec(wv, (c, 1, 3), &dev)?;
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

    /// 2d cases: (name, b, c, h, w, k_h, k_w, padding, stride, dilation)
    #[allow(clippy::type_complexity)]
    const CASES_2D: &[(
        &str,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    )] = &[
        ("mobilenet-3x3", 2, 32, 14, 14, 3, 3, 1, 1, 1),
        ("mobilenet-3x3-s2", 1, 16, 15, 15, 3, 3, 1, 2, 1),
        ("small-odd", 1, 8, 7, 6, 2, 3, 0, 1, 1),
        ("k1", 1, 4, 5, 5, 1, 1, 0, 1, 1),
        ("dilation", 1, 9, 12, 11, 3, 3, 2, 1, 2),
        ("stride3", 2, 9, 13, 12, 3, 3, 1, 3, 1),
    ];

    #[test]
    fn depthwise_conv2d_matches_grouped_loop() -> Result<()> {
        for (dev_name, dev) in devices() {
            let mut rng = Lcg::new(77);
            for &(name, b, c, i_h, i_w, k_h, k_w, padding, stride, dilation) in CASES_2D {
                let x = rng.tensor((b, c, i_h, i_w), &dev)?;
                let k = rng.tensor((c, 1, k_h, k_w), &dev)?;
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
                assert!(
                    depthwise_2d_applicable(&params, c),
                    "{name}: expected the fast path to be applicable"
                );
                let want = x.conv2d_grouped_loop(&k, &params, c)?;
                let got = x.conv2d_depthwise(&k, &params)?;
                assert_eq!(got.dims(), want.dims(), "{dev_name}/{name}: shape");
                let diff = max_abs_diff(&got, &want)?;
                assert!(diff <= 1e-5, "{dev_name}/{name}: max abs diff {diff}");
                let public = x.conv2d(&k, padding, stride, dilation, c)?;
                let diff = max_abs_diff(&public, &want)?;
                assert!(
                    diff <= 1e-5,
                    "{dev_name}/{name}: public max abs diff {diff}"
                );
                let brute = brute_force_2d(&x, &k, &params)?;
                let diff = max_abs_diff(&got, &brute)?;
                assert!(
                    diff <= 1e-4,
                    "{dev_name}/{name}: brute force max abs diff {diff}"
                );
            }
        }
        Ok(())
    }
}
