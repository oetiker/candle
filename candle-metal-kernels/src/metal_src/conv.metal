#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))

template <typename T>
METAL_FUNC void im2col(
    constant size_t &dst_numel,
    constant size_t &h_out,
    constant size_t &w_out,
    constant size_t &h_k,
    constant size_t &w_k,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &dilation,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  // dst: (b_size, h_out, w_out, c_in, h_k, w_k)
  // src: (b_size, c_in, h_in, w_in)
  if (tid >= dst_numel) {
    return;
  }
  // The whole destination decode is done in 32 bits. `tid` is a `uint`, so no destination index or
  // stride can exceed it, and 64-bit integer division is emulated in software on this GPU: five of
  // them per output element made im2col ALU-bound at ~7 GB/s, while a plain elementwise pass over
  // the same tensor sustains ~100 GB/s. Only the source index, which is scaled by caller-supplied
  // strides, stays 64-bit. Measured on ReDimNet-b6: the grouped 3x3 convolution of stage 0 drops
  // from 270 ms to 63 ms, and a full forward from 13.90 s to 9.38 s.
  const uint c_in = uint(src_dims[1]);
  const size_t h_in = src_dims[2];
  const size_t w_in = src_dims[3];

  const uint dst_s4 = uint(w_k);
  const uint dst_s3 = uint(h_k) * dst_s4;
  const uint dst_s2 = c_in * dst_s3;
  const uint dst_s1 = uint(w_out) * dst_s2;
  const uint dst_s0 = uint(h_out) * dst_s1;

  uint tmp_tid = tid;
  const uint b_idx = tmp_tid / dst_s0;
  tmp_tid -= b_idx * dst_s0;
  const uint h_idx = tmp_tid / dst_s1;
  tmp_tid -= h_idx * dst_s1;
  const uint w_idx = tmp_tid / dst_s2;
  tmp_tid -= w_idx * dst_s2;
  const uint c_idx = tmp_tid / dst_s3;
  tmp_tid -= c_idx * dst_s3;
  const uint h_k_idx = tmp_tid / dst_s4;
  const uint w_k_idx = tmp_tid - h_k_idx * dst_s4;
  size_t src_h_idx = h_idx * stride + h_k_idx * dilation;
  size_t src_w_idx = w_idx * stride + w_k_idx * dilation;
  if (src_h_idx < padding || src_h_idx >= h_in + padding) {
    dst[tid] = static_cast<T>(0);
  }
  else if (src_w_idx < padding || src_w_idx >= w_in + padding) {
    dst[tid] = static_cast<T>(0);
  }
  else {
    src_h_idx -= padding;
    src_w_idx -= padding;
    const size_t src_i =
      b_idx * src_strides[0]
      + c_idx * src_strides[1]
      + src_h_idx * src_strides[2]
      + src_w_idx * src_strides[3];
    dst[tid] = src[src_i];
  }
}

template <typename T>
METAL_FUNC void col2im1d(
    constant size_t &dst_el,
    constant size_t &l_out,
    constant size_t &l_in,
    constant size_t &c_out,
    constant size_t &k_size,
    constant size_t &stride,
    device const T *src,
    device T *dst,
    uint dst_i [[ thread_position_in_grid ]]
) {
  // src: (b_size, l_in, c_out, l_k)
  // dst: (b_size, c_out, l_out)
  if (dst_i >= dst_el) {
    return;
  }

  const size_t dst_s0 = c_out * l_out;
  const size_t dst_s1 = l_out;
  const size_t src_s0 = c_out * k_size * l_in;
  const size_t src_s1 = c_out * k_size;
  const size_t src_s2 = k_size;

  size_t tmp_dst_i = dst_i;
  const size_t b_idx = tmp_dst_i / dst_s0;
  tmp_dst_i -= b_idx * dst_s0;
  const size_t c_idx = tmp_dst_i / dst_s1;
  tmp_dst_i -= c_idx * dst_s1;
  const int l_out_idx = tmp_dst_i;

  dst[dst_i] = static_cast<T>(0);

  int l_in_idx = l_out_idx / stride;
  int k0 = l_out_idx - l_in_idx * stride;
  // l_out_idx = l_in_idx * stride + k0
  for (; k0 < k_size && l_in_idx >= 0; k0 += stride, --l_in_idx) {
    if (l_in_idx < l_in) {
      const size_t src_i = b_idx * src_s0 + l_in_idx * src_s1 + c_idx * src_s2 + k0;
      dst[dst_i] += src[src_i];
    }
  }
}

template <typename T>
METAL_FUNC void im2col1d(
    constant size_t &dst_numel,
    constant size_t &l_out,
    constant size_t &l_k,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &dilation,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  // dst: (b_size, l_out, c_in, l_k)
  // src: (b_size, c_in, l_in)
  if (tid >= dst_numel) {
    return;
  }
  // 32-bit destination decode, for the reason spelled out in `im2col` above: `tid` is a `uint`, so
  // the indices cannot overflow it, and 64-bit integer division is emulated in software here.
  const uint c_in = uint(src_dims[1]);
  const size_t l_in = src_dims[2];

  const uint dst_s2 = uint(l_k);
  const uint dst_s1 = c_in * dst_s2;
  const uint dst_s0 = uint(l_out) * dst_s1;

  uint tmp_dst_i = tid;
  const uint b_idx = tmp_dst_i / dst_s0;
  tmp_dst_i -= b_idx * dst_s0;
  const uint l_idx = tmp_dst_i / dst_s1;
  tmp_dst_i -= l_idx * dst_s1;
  const uint c_idx = tmp_dst_i / dst_s2;
  const uint l_k_idx = tmp_dst_i - c_idx * dst_s2;
  size_t src_l_idx = l_idx * stride + l_k_idx * dilation;
  if (src_l_idx < padding || src_l_idx >= l_in + padding) {
    dst[tid] = static_cast<T>(0);
  }
  else {
    src_l_idx -= padding;
    const size_t src_i = b_idx * src_strides[0] + c_idx * src_strides[1] + src_l_idx * src_strides[2];
    dst[tid] = src[src_i];
  }
}

template <typename T>
METAL_FUNC void upsample_nearest2d(
    constant size_t &w_out,
    constant size_t &h_out,
    constant float &w_scale,
    constant float &h_scale,
    constant size_t *src_dims,
    constant size_t *src_s,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  // src: (b_size, c_in, w_in, h_in)

  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  if (tid >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  // TODO: Improve this.
  const size_t b_idx = tid / (w_out * h_out * c);
  const size_t c_idx = (tid / (w_out * h_out)) % c;
  const size_t dst_w = (tid / h_out) % w_out;
  const size_t dst_h = tid % h_out;

  size_t src_w = static_cast<size_t>(dst_w * w_scale);
  size_t src_h = static_cast<size_t>(dst_h * h_scale);
  if (src_w >= w_in) {
    src_w = w_in - 1;
  }
  if (src_h >= h_in) {
    src_h = h_in - 1;
  }

  const size_t src_i = b_idx * src_s[0] + c_idx * src_s[1] + src_w * src_s[2] + src_h * src_s[3];
  dst[tid] = src[src_i];
}

template <typename T>
METAL_FUNC void upsample_bilinear2d(
    constant size_t &w_out,
    constant size_t &h_out,
    constant bool &align_corners,
    constant bool &has_scale_h,
    constant float &scale_h_factor,
    constant bool &has_scale_w,
    constant float &scale_w_factor,
    constant size_t *src_dims,
    constant size_t *src_s,
    device const T *src,
    device T *dst,
    uint tid [[thread_position_in_grid]]
) {
    // src: (b_size, c_in, h_in, w_in)  // Standard NCHW layout
    const size_t c = src_dims[1];
    const size_t h_in = src_dims[2];  // dims[2] = height
    const size_t w_in = src_dims[3];  // dims[3] = width
    
    if (tid >= src_dims[0] * c * h_out * w_out) {
        return;
    }
    
    // Compute output position (NCHW layout)
    const size_t b_idx = tid / (h_out * w_out * c);
    const size_t c_idx = (tid / (h_out * w_out)) % c;
    const size_t dst_h = (tid / w_out) % h_out;
    const size_t dst_w = tid % w_out;
    
    // Calculate scale factors following PyTorch's area_pixel_compute_scale logic
    float h_scale, w_scale;
    if (align_corners) {
        h_scale = (h_out > 1) ? static_cast<float>(h_in - 1) / (h_out - 1) : 0.0f;
        w_scale = (w_out > 1) ? static_cast<float>(w_in - 1) / (w_out - 1) : 0.0f;
    } else {
        // PyTorch's compute_scales_value logic
        h_scale = has_scale_h ? (1.0f / scale_h_factor) : (static_cast<float>(h_in) / h_out);
        w_scale = has_scale_w ? (1.0f / scale_w_factor) : (static_cast<float>(w_in) / w_out);
    }
    
    // Compute source position
    float src_h_fp, src_w_fp;
    if (align_corners) {
        src_h_fp = h_scale * dst_h;
        src_w_fp = w_scale * dst_w;
    } else {
        src_h_fp = h_scale * (dst_h + 0.5f) - 0.5f;
        src_w_fp = w_scale * (dst_w + 0.5f) - 0.5f;
    }
    
    // Clamp to valid range
    src_h_fp = max(0.0f, src_h_fp);
    src_w_fp = max(0.0f, src_w_fp);
    
    // Get integer indices
    size_t h0 = static_cast<size_t>(floor(src_h_fp));
    size_t w0 = static_cast<size_t>(floor(src_w_fp));
    size_t h1 = min(h0 + 1, h_in - 1);
    size_t w1 = min(w0 + 1, w_in - 1);
    
    // Compute interpolation weights
    float weight_h = src_h_fp - h0;
    float weight_w = src_w_fp - w0;
    weight_h = clamp(weight_h, 0.0f, 1.0f);
    weight_w = clamp(weight_w, 0.0f, 1.0f);
    
    // Get base index
    const size_t base = b_idx * src_s[0] + c_idx * src_s[1];
    
    // Read four neighboring pixels
    const T v00 = src[base + h0 * src_s[2] + w0 * src_s[3]];
    const T v10 = src[base + h0 * src_s[2] + w1 * src_s[3]];
    const T v01 = src[base + h1 * src_s[2] + w0 * src_s[3]];
    const T v11 = src[base + h1 * src_s[2] + w1 * src_s[3]];
    
    // Bilinear interpolation
    const float v_top = float(v00) * (1.0f - weight_w) + float(v10) * weight_w;
    const float v_bottom = float(v01) * (1.0f - weight_w) + float(v11) * weight_w;
    const float value = v_top * (1.0f - weight_h) + v_bottom * weight_h;
    
    dst[tid] = T(value);
}

#define IM2COL_OP(T, FN_NAME) \
kernel void FN_NAME(  \
    constant size_t &dst_numel, \
    constant size_t &h_out, \
    constant size_t &w_out, \
    constant size_t &h_k, \
    constant size_t &w_k, \
    constant size_t &stride, \
    constant size_t &padding, \
    constant size_t &dilation, \
    constant size_t *src_dims, \
    constant size_t *src_strides, \
    device const T *src, \
    device T *dst, \
    uint tid [[ thread_position_in_grid ]] \
) {  \
  im2col<T>(dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, src_dims, src_strides, src, dst, tid); \
} \

#define IM2COL1D_OP(T, FN_NAME) \
kernel void FN_NAME(  \
    constant size_t &dst_numel, \
    constant size_t &l_out, \
    constant size_t &l_k, \
    constant size_t &stride, \
    constant size_t &padding, \
    constant size_t &dilation, \
    constant size_t *src_dims, \
    constant size_t *src_strides, \
    device const T *src, \
    device T *dst, \
    uint tid [[ thread_position_in_grid ]] \
) {  \
  im2col1d<T>(dst_numel, l_out, l_k, stride, padding, dilation, src_dims, src_strides, src, dst, tid); \
} \

#define COL2IM1D_OP(T, FN_NAME) \
kernel void FN_NAME(  \
    constant size_t &dst_el, \
    constant size_t &l_out, \
    constant size_t &l_in, \
    constant size_t &c_out, \
    constant size_t &k_size, \
    constant size_t &stride, \
    device const T *src, \
    device T *dst, \
    uint tid [[ thread_position_in_grid ]] \
) {  \
  col2im1d<T>(dst_el, l_out, l_in, c_out, k_size, stride, src, dst, tid); \
} \

#define UPSAMPLE_NEAREST2D_OP(TYPENAME, FN_NAME) \
kernel void FN_NAME(  \
    constant size_t &w_out, \
    constant size_t &h_out, \
    constant float &w_scale, \
    constant float &h_scale, \
    constant size_t *dims, \
    constant size_t *strides, \
    device const TYPENAME *src, \
    device TYPENAME *dst, \
    uint tid [[ thread_position_in_grid ]] \
) {  \
  upsample_nearest2d<TYPENAME>(w_out, h_out, w_scale, h_scale, dims, strides, src, dst, tid); \
} \

#define UPSAMPLE_BILINEAR2D_OP(TYPENAME, FN_NAME) \
kernel void FN_NAME(  \
    constant size_t &w_out [[buffer(0)]], \
    constant size_t &h_out [[buffer(1)]], \
    constant bool &align_corners [[buffer(2)]], \
    constant bool &has_scale_h [[buffer(3)]], \
    constant float &scale_h_factor [[buffer(4)]], \
    constant bool &has_scale_w [[buffer(5)]], \
    constant float &scale_w_factor [[buffer(6)]], \
    constant size_t *src_dims [[buffer(7)]], \
    constant size_t *src_s [[buffer(8)]], \
    device const TYPENAME *src [[buffer(9)]], \
    device TYPENAME *dst [[buffer(10)]], \
    uint tid [[thread_position_in_grid]] \
) {  \
  upsample_bilinear2d<TYPENAME>(w_out, h_out, align_corners, has_scale_h, scale_h_factor, has_scale_w, scale_w_factor, src_dims, src_s, src, dst, tid); \
} \

template <typename T, typename A>
METAL_FUNC void avg_pool2d(
    constant size_t &w_k,
    constant size_t &h_k,
    constant size_t &w_stride,
    constant size_t &h_stride,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  const size_t w_out = (w_in - w_k) / w_stride + 1;
  const size_t h_out = (h_in - h_k) / h_stride + 1;
  if (tid >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  const size_t b_idx = tid / (w_out * h_out * c);
  const size_t c_idx = (tid / (w_out * h_out)) % c;
  const size_t dst_w = (tid / h_out) % w_out;
  const size_t dst_h = tid % h_out;

  const size_t src_idx0 = b_idx * src_strides[0];
  A d = 0;
  for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
    size_t src_w = w_stride * dst_w + w_offset;
    if (src_w >= w_in){
      continue;
    }
    for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
      size_t src_h = h_stride * dst_h + h_offset;
      if (src_h >= h_in) {
        continue;
      }
      const size_t src_idx = src_idx0 + c_idx * src_strides[1] + src_w * src_strides[2] + src_h * src_strides[3];
      d += static_cast<A>(src[src_idx]);
    }
  }
  dst[tid] = static_cast<T>(d / (w_k * h_k));
}

#define AVGPOOL2D_OP(TYPENAME, TYPEACC, FN_NAME) \
kernel void FN_NAME( \
    constant size_t &w_k, \
    constant size_t &h_k, \
    constant size_t &w_s, \
    constant size_t &h_s, \
    constant size_t *src_dims, \
    constant size_t *src_s, \
    device const TYPENAME *src, \
    device TYPENAME *dst, \
    uint tid [[ thread_position_in_grid ]] \
) { \
  avg_pool2d<TYPENAME, TYPEACC>(w_k, h_k, w_s, h_s, src_dims, src_s, src, dst, tid); \
} \

template <typename T>
METAL_FUNC void max_pool2d(
    constant size_t &w_k,
    constant size_t &h_k,
    constant size_t &w_stride,
    constant size_t &h_stride,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  const size_t w_out = (w_in - w_k) / w_stride + 1;
  const size_t h_out = (h_in - h_k) / h_stride + 1;
  if (tid >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  const size_t b_idx = tid / (w_out * h_out * c);
  const size_t c_idx = (tid / (w_out * h_out)) % c;
  const size_t dst_w = (tid / h_out) % w_out;
  const size_t dst_h = tid % h_out;

  const size_t src_idx0 = b_idx * src_strides[0];
  T d = 0;
  bool set = false;
  for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
    size_t src_w = w_stride * dst_w + w_offset;
    if (src_w >= w_in){
      continue;
    }
    for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
      size_t src_h = h_stride * dst_h + h_offset;
      if (src_h >= h_in) {
        continue;
      }
      const size_t src_idx = src_idx0 + c_idx * src_strides[1] + src_w * src_strides[2] + src_h * src_strides[3];
      if (set) {
        d = MAX(d, src[src_idx]);
      }
      else {
        d = src[src_idx];
        set = true;
      }
    }
  }
  dst[tid] = d;
}

#define MAXPOOL2D_OP(TYPENAME, FN_NAME) \
kernel void FN_NAME( \
    constant size_t &w_k, \
    constant size_t &h_k, \
    constant size_t &w_s, \
    constant size_t &h_s, \
    constant size_t *src_dims, \
    constant size_t *src_s, \
    device const TYPENAME *src, \
    device TYPENAME *dst, \
    uint tid [[ thread_position_in_grid ]] \
) { \
  max_pool2d<TYPENAME>(w_k, h_k, w_s, h_s, src_dims, src_s, src, dst, tid); \
} \


// Naive implementation of conv_transpose1d.
template <typename T, typename A>
METAL_FUNC void conv_transpose1d(
    constant size_t &l_out,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &out_padding,
    constant size_t &dilation,
    constant size_t *src_dims,
    constant size_t *src_strides,
    constant size_t *k_dims,
    constant size_t *k_strides,
    device const T *src,
    device const T *k,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  // src: (b_size, c_in, l_in)
  // kernel: (c_in, c_out, l_k)
  const size_t l_k = k_dims[2];
  const size_t c_out = k_dims[1];
  const size_t c_in = src_dims[1];
  const size_t l_in = src_dims[2];
  if (tid >= src_dims[0] * c_out * l_out) {
    return;
  }

  const size_t b_idx = tid / (l_out * c_out);
  const size_t dst_c_idx = (tid / l_out) % c_out;
  const size_t out_x = tid % l_out;

  const size_t src_idx0 = b_idx * src_strides[0];
  A d = 0;
  for (int k_x = 0; k_x < (int)l_k; ++k_x) {
      // let out_x = inp_x * p.stride + k_x * p.dilation - p.padding;
      int inp_x_stride = (int)(out_x + padding) - k_x * dilation;
      if (inp_x_stride < 0 || inp_x_stride % stride) {
          continue;
      }
      int inp_x = inp_x_stride / stride;
      if (inp_x >= l_in) continue;
      for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
          const size_t src_idx = src_idx0 + src_c_idx * src_strides[1] + inp_x * src_strides[2];
          const size_t k_idx = src_c_idx * k_strides[0] + dst_c_idx * k_strides[1] + k_x * k_strides[2];
          d += static_cast<A>(src[src_idx]) * static_cast<A>(k[k_idx]);
      }
  }
  dst[tid] = static_cast<T>(d);
}

#define CONVT1D_OP(TYPENAME, TYPEACC, FN_NAME) \
kernel void FN_NAME(  \
    constant size_t &l_out, \
    constant size_t &stride, \
    constant size_t &padding, \
    constant size_t &out_padding, \
    constant size_t &dilation, \
    constant size_t *src_dims, \
    constant size_t *src_strides, \
    constant size_t *k_dims, \
    constant size_t *k_strides, \
    device const TYPENAME *src, \
    device const TYPENAME *k, \
    device TYPENAME *dst, \
    uint tid [[ thread_position_in_grid ]] \
) {  \
  conv_transpose1d<TYPENAME, TYPEACC>(l_out, stride, padding, out_padding, dilation, src_dims, src_strides, k_dims, k_strides, src, k, dst, tid); \
} \

template <typename T, typename A>
METAL_FUNC void conv_transpose2d(
  constant size_t &w_out,
  constant size_t &h_out,
  constant size_t &stride,
  constant size_t &padding,
  constant size_t &out_padding,
  constant size_t &dilation,
  constant size_t *input_dims,
  constant size_t *input_stride,
  constant size_t *k_dims,
  constant size_t *k_stride,
  device const T *src,
  device const T *k,
  device T *dst,
  uint tid [[ thread_position_in_grid ]]
) {
  const size_t h_k = k_dims[2];
  const size_t w_k = k_dims[3];
  const size_t c_out = k_dims[1];
  const size_t c_in = input_dims[1];
  const size_t h_in = input_dims[2];
  const size_t w_in = input_dims[3];

  if (tid >= input_dims[0] * c_out * w_out * h_out) {
    return;
  }

  const size_t b_idx = tid / (w_out * h_out * c_out);
  const size_t dst_c_idx = (tid / (w_out * h_out)) % c_out;
  const size_t out_y = (tid / w_out) % h_out;
  const size_t out_x = tid % w_out;

  const size_t src_idx0 = b_idx * input_stride[0];

  A d = 0;
  for (int k_x = 0; k_x < (int)w_k; ++k_x) {
      const int inp_x_stride = (int)(out_x + padding) - k_x * dilation;
      if (inp_x_stride < 0 || inp_x_stride % stride) {
          continue;
      }
      const int inp_x = inp_x_stride / stride;
      if (inp_x >= w_in) continue;
      for (int k_y = 0; k_y < (int)h_k; ++k_y) {
          const int inp_y_stride = (int)(out_y + padding) - k_y * dilation;
          if (inp_y_stride < 0 || inp_y_stride % stride) {
              continue;
          }
          const int inp_y = inp_y_stride / stride;
          if (inp_y >= h_in) continue;
          for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
              const size_t src_idx = src_idx0 + src_c_idx * input_stride[1] + inp_y * input_stride[2] + inp_x * input_stride[3];
              const size_t k_idx = src_c_idx * k_stride[0] + dst_c_idx * k_stride[1] + k_y * k_stride[2] + k_x * k_stride[3];
              d += static_cast<A>(src[src_idx]) * static_cast<A>(k[k_idx]);
          }
      }
  }
  dst[tid] = static_cast<T>(d);
}

#define CONVT2D_OP(TYPENAME, TYPEACC, FN_NAME) \
kernel void FN_NAME(  \
    constant size_t &w_out, \
    constant size_t &h_out, \
    constant size_t &stride, \
    constant size_t &padding, \
    constant size_t &out_padding, \
    constant size_t &dilation, \
    constant size_t *input_dims, \
    constant size_t *input_stride, \
    constant size_t *k_dims, \
    constant size_t *k_stride, \
    device const TYPENAME *src, \
    device const TYPENAME *k, \
    device TYPENAME *dst, \
    uint tid [[ thread_position_in_grid ]] \
) {  \
  conv_transpose2d<TYPENAME, TYPEACC>(w_out, h_out, stride, padding, out_padding, dilation, input_dims, input_stride, k_dims, k_stride, src, k, dst, tid); \
} \

template <typename T>
METAL_FUNC void conv2d_grouped_direct(
    constant size_t &dst_numel,
    constant size_t &groups,
    constant size_t &c_in_pg,
    constant size_t &c_out_pg,
    constant size_t &h_in,
    constant size_t &w_in,
    constant size_t &h_out,
    constant size_t &w_out,
    constant size_t &k_h,
    constant size_t &k_w,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &dilation,
    device const T *src,
    device const T *weight,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  // src:    (b, groups * c_in_pg, h_in, w_in), contiguous
  // weight: (groups * c_out_pg, c_in_pg, k_h, k_w), contiguous
  // dst:    (b, groups * c_out_pg, h_out, w_out), contiguous
  //
  // One thread per OUTPUT ELEMENT. Adjacent threads differ in the width index, so both the source
  // reads and the destination write are coalesced, and every thread in a threadgroup shares the
  // same weights, which the cache broadcasts. im2col is never materialised: the gather is the
  // indexing below.
  if (tid >= dst_numel) {
    return;
  }
  // 32-bit decode. `tid` is a `uint` so no destination index can exceed it, and 64-bit integer
  // division is emulated in software on this GPU -- the same defect that cost 4.3x in im2col.
  const uint w_out_u = uint(w_out);
  const uint h_out_u = uint(h_out);
  const uint c_out_pg_u = uint(c_out_pg);
  const uint c_in_pg_u = uint(c_in_pg);
  const uint c_out_u = c_out_pg_u * uint(groups);

  uint rest = tid;
  const uint x_idx = rest % w_out_u;
  rest /= w_out_u;
  const uint y_idx = rest % h_out_u;
  rest /= h_out_u;
  const uint co = rest % c_out_u;
  const uint b_idx = rest / c_out_u;
  const uint g = co / c_out_pg_u;

  const size_t plane = h_in * w_in;
  const size_t src_base =
    (size_t)b_idx * (size_t)(c_in_pg_u * uint(groups)) * plane + (size_t)(g * c_in_pg_u) * plane;
  const size_t w_base = (size_t)co * c_in_pg * k_h * k_w;

  // Accumulate in float even for narrower T: the summation order already differs from the
  // im2col+GEMM path, so there is no reason to also lose precision to the accumulator.
  float acc = 0.0f;
  for (uint ci = 0; ci < c_in_pg_u; ++ci) {
    const size_t src_c = src_base + (size_t)ci * plane;
    const size_t w_c = w_base + (size_t)ci * k_h * k_w;
    for (uint ky = 0; ky < uint(k_h); ++ky) {
      const long iy = (long)y_idx * (long)stride + (long)ky * (long)dilation - (long)padding;
      if (iy < 0 || iy >= (long)h_in) {
        continue;
      }
      for (uint kx = 0; kx < uint(k_w); ++kx) {
        const long ix = (long)x_idx * (long)stride + (long)kx * (long)dilation - (long)padding;
        if (ix < 0 || ix >= (long)w_in) {
          continue;
        }
        acc += float(src[src_c + (size_t)iy * w_in + (size_t)ix])
             * float(weight[w_c + (size_t)ky * k_w + (size_t)kx]);
      }
    }
  }
  dst[tid] = static_cast<T>(acc);
}

#define CONV2D_GROUPED_DIRECT_OP(T, FN_NAME) \
kernel void FN_NAME( \
    constant size_t &dst_numel, \
    constant size_t &groups, \
    constant size_t &c_in_pg, \
    constant size_t &c_out_pg, \
    constant size_t &h_in, \
    constant size_t &w_in, \
    constant size_t &h_out, \
    constant size_t &w_out, \
    constant size_t &k_h, \
    constant size_t &k_w, \
    constant size_t &stride, \
    constant size_t &padding, \
    constant size_t &dilation, \
    device const T *src, \
    device const T *weight, \
    device T *dst, \
    uint tid [[ thread_position_in_grid ]] \
) { \
  conv2d_grouped_direct<T>(dst_numel, groups, c_in_pg, c_out_pg, h_in, w_in, h_out, w_out, \
                           k_h, k_w, stride, padding, dilation, src, weight, dst, tid); \
} \

// Implicit GEMM: out[c_out_pg, P] = W[c_out_pg, c_in_pg * 9] x Xcol[.., P], where Xcol is gathered
// into threadgroup memory and NEVER written to device memory. One threadgroup per
// (batch * group, output row, tile of TILE_T output columns).
//
// Restricted, by the dispatch predicate, to: stride 1, dilation 1, k_h == k_w == 3, padding 1,
// c_out_pg == 32, c_in_pg a multiple of CI_CHUNK. Anything else uses the simple kernel.
//
// Where the reuse comes from. Each thread owns a REGISTER TILE of CO_REG output channels by T_REG
// output columns. For every (input channel, ky, kx) it loads CO_REG weights and T_REG activations
// out of threadgroup memory and issues CO_REG * T_REG fused multiply-adds against them, so the
// ratio of arithmetic to threadgroup traffic is CO_REG * T_REG / (CO_REG + T_REG) -- 4.0 at
// 8x8, against 9/45 for a kernel that keeps no register tile at all. The weight slab is read from
// device memory once per threadgroup instead of once per output element, which is what the simple
// one-thread-per-output kernel does.
//
// The T_REG columns a thread owns are INTERLEAVED (column t_blk + j * NT, not t_blk * T_REG + j) so
// that lane-adjacent threads touch adjacent threadgroup words and adjacent destination addresses.
// A contiguous per-thread run would make the threadgroup stride T_REG words and serialise the
// loads across banks.
template <typename T, uint TILE_T, uint CI_CHUNK, uint CO_REG, uint T_REG>
METAL_FUNC void conv2d_grouped_tiled(
    constant size_t &groups,
    constant size_t &c_in_pg,
    constant size_t &h_in,
    constant size_t &w_in,
    device const T *src,
    device const T *weight,
    device T *dst,
    threadgroup float *sx,
    threadgroup float *sw,
    uint3 tgid,
    uint tix
) {
  const uint TILE_CO = 32;                  // == c_out_pg, enforced by the predicate
  const uint SX_ROW = TILE_T + 2;           // one staged input row, halo included
  const uint NT = TILE_T / T_REG;           // column blocks == threads per channel block
  const uint threads = NT * (TILE_CO / CO_REG);

  const uint t_blk = tix % NT;
  const uint co0 = (tix / NT) * CO_REG;

  const uint groups_u = uint(groups);
  const uint g = tgid.z % groups_u;
  const uint b_idx = tgid.z / groups_u;
  const uint y_out = tgid.y;
  const uint x0 = tgid.x * TILE_T;

  const uint h_in_u = uint(h_in);
  const uint w_in_u = uint(w_in);
  const uint c_in_pg_u = uint(c_in_pg);

  float acc[CO_REG][T_REG];
  for (uint r = 0; r < CO_REG; ++r) {
    for (uint j = 0; j < T_REG; ++j) {
      acc[r][j] = 0.0f;
    }
  }

  const size_t plane = (size_t)h_in_u * (size_t)w_in_u;
  const size_t src_g =
    ((size_t)b_idx * (size_t)(c_in_pg_u * groups_u) + (size_t)(g * c_in_pg_u)) * plane;
  const size_t w_g = (size_t)(g * TILE_CO) * (size_t)c_in_pg_u * 9;

  for (uint ci0 = 0; ci0 < c_in_pg_u; ci0 += CI_CHUNK) {
    // --- stage the input tile: CI_CHUNK channels x 3 rows x (TILE_T + 2) columns ---
    // sx[((sc * 3) + sr) * SX_ROW + scol] holds input column x0 + scol - 1, row y_out + sr - 1.
    for (uint slot = tix; slot < CI_CHUNK * 3 * SX_ROW; slot += threads) {
      const uint sc = slot / (3 * SX_ROW);
      const uint rem = slot - sc * (3 * SX_ROW);
      const uint sr = rem / SX_ROW;
      const uint scol = rem - sr * SX_ROW;
      const int iy = int(y_out) + int(sr) - 1;   // padding == 1
      const int ix = int(x0) + int(scol) - 1;
      float v = 0.0f;
      if (iy >= 0 && iy < int(h_in_u) && ix >= 0 && ix < int(w_in_u)) {
        v = float(src[src_g + (size_t)(ci0 + sc) * plane
                    + (size_t)uint(iy) * (size_t)w_in_u + (size_t)uint(ix)]);
      }
      sx[slot] = v;
    }
    // --- stage the weight slab, laid out [ci][tap][c_out] so a whole channel block of weights is
    // contiguous and lane-uniform in the inner loop ---
    for (uint slot = tix; slot < CI_CHUNK * 9 * TILE_CO; slot += threads) {
      const uint sc = slot / (9 * TILE_CO);
      const uint rem = slot - sc * (9 * TILE_CO);
      const uint tap = rem / TILE_CO;
      const uint sco = rem - tap * TILE_CO;
      sw[slot] = float(weight[w_g + (size_t)sco * (size_t)c_in_pg_u * 9
                            + (size_t)(ci0 + sc) * 9 + (size_t)tap]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- the register-tiled inner product ---
    for (uint sc = 0; sc < CI_CHUNK; ++sc) {
      for (uint ky = 0; ky < 3; ++ky) {
        for (uint kx = 0; kx < 3; ++kx) {
          float wv[CO_REG];
          for (uint r = 0; r < CO_REG; ++r) {
            wv[r] = sw[(sc * 9 + ky * 3 + kx) * TILE_CO + co0 + r];
          }
          float xv[T_REG];
          for (uint j = 0; j < T_REG; ++j) {
            xv[j] = sx[(sc * 3 + ky) * SX_ROW + t_blk + j * NT + kx];
          }
          for (uint r = 0; r < CO_REG; ++r) {
            for (uint j = 0; j < T_REG; ++j) {
              acc[r][j] += wv[r] * xv[j];
            }
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // stride 1 and padding 1 with a 3x3 window => h_out == h_in and w_out == w_in.
  const size_t dst_base =
    ((size_t)b_idx * (size_t)(TILE_CO * groups_u) + (size_t)(g * TILE_CO)) * plane
    + (size_t)y_out * (size_t)w_in_u;
  for (uint j = 0; j < T_REG; ++j) {
    const uint x_out = x0 + t_blk + j * NT;
    if (x_out >= w_in_u) {
      continue;
    }
    for (uint r = 0; r < CO_REG; ++r) {
      dst[dst_base + (size_t)(co0 + r) * plane + (size_t)x_out] = static_cast<T>(acc[r][j]);
    }
  }
}

// The threadgroup arrays are declared in the kernel entry point: Metal only allows threadgroup
// address space variables at kernel scope, so they are passed down as pointers.
#define CONV2D_GROUPED_TILED_OP(T, TILE_T, CI_CHUNK, CO_REG, T_REG, FN_NAME) \
kernel void FN_NAME( \
    constant size_t &groups, \
    constant size_t &c_in_pg, \
    constant size_t &h_in, \
    constant size_t &w_in, \
    device const T *src, \
    device const T *weight, \
    device T *dst, \
    uint3 tgid [[ threadgroup_position_in_grid ]], \
    uint tix [[ thread_index_in_threadgroup ]] \
) { \
  threadgroup float sx[CI_CHUNK * 3 * (TILE_T + 2)]; \
  threadgroup float sw[CI_CHUNK * 9 * 32]; \
  conv2d_grouped_tiled<T, TILE_T, CI_CHUNK, CO_REG, T_REG>( \
    groups, c_in_pg, h_in, w_in, src, weight, dst, sx, sw, tgid, tix); \
} \

IM2COL_OP(float, im2col_f32)
IM2COL_OP(half, im2col_f16)
IM2COL_OP(uint8_t, im2col_u8)
IM2COL_OP(uint32_t, im2col_u32)
#if defined(__HAVE_BFLOAT__)
IM2COL_OP(bfloat, im2col_bf16)
#endif

COL2IM1D_OP(float, col2im1d_f32)
COL2IM1D_OP(half, col2im1d_f16)
COL2IM1D_OP(uint8_t, col2im1d_u8)
COL2IM1D_OP(uint32_t, col2im1d_u32)
#if defined(__HAVE_BFLOAT__)
COL2IM1D_OP(bfloat, col2im1d_bf16)
#endif

IM2COL1D_OP(float, im2col1d_f32)
IM2COL1D_OP(half, im2col1d_f16)
IM2COL1D_OP(uint8_t, im2col1d_u8)
IM2COL1D_OP(uint32_t, im2col1d_u32)
#if defined(__HAVE_BFLOAT__)
IM2COL1D_OP(bfloat, im2col1d_bf16)
#endif

UPSAMPLE_NEAREST2D_OP(float, upsample_nearest2d_f32)
UPSAMPLE_NEAREST2D_OP(half, upsample_nearest2d_f16)
UPSAMPLE_NEAREST2D_OP(uint8_t, upsample_nearest2d_u8)
UPSAMPLE_NEAREST2D_OP(uint32_t, upsample_nearest2d_u32)
#if defined(__HAVE_BFLOAT__)
UPSAMPLE_NEAREST2D_OP(bfloat, upsample_nearest2d_bf16)
#endif

UPSAMPLE_BILINEAR2D_OP(float, upsample_bilinear2d_f32)
UPSAMPLE_BILINEAR2D_OP(half, upsample_bilinear2d_f16)
UPSAMPLE_BILINEAR2D_OP(uint8_t, upsample_bilinear2d_u8)
UPSAMPLE_BILINEAR2D_OP(uint32_t, upsample_bilinear2d_u32)
#if defined(__HAVE_BFLOAT__)
UPSAMPLE_BILINEAR2D_OP(bfloat, upsample_bilinear2d_bf16)
#endif

MAXPOOL2D_OP(float, max_pool2d_f32)
MAXPOOL2D_OP(half, max_pool2d_f16)
MAXPOOL2D_OP(uint32_t, max_pool2d_u32)
MAXPOOL2D_OP(uint8_t, max_pool2d_u8)
#if defined(__HAVE_BFLOAT__)
MAXPOOL2D_OP(bfloat, max_pool2d_bf16)
#endif

AVGPOOL2D_OP(float, float, avg_pool2d_f32)
AVGPOOL2D_OP(half, float, avg_pool2d_f16)
AVGPOOL2D_OP(uint32_t, uint32_t, avg_pool2d_u32)
AVGPOOL2D_OP(uint8_t, uint8_t, avg_pool2d_u8)
#if defined(__HAVE_BFLOAT__)
AVGPOOL2D_OP(bfloat, float, avg_pool2d_bf16)
#endif

CONVT1D_OP(float, float, conv_transpose1d_f32)
CONVT1D_OP(half, float, conv_transpose1d_f16)
CONVT1D_OP(uint8_t, uint8_t, conv_transpose1d_u8)
CONVT1D_OP(uint32_t, uint32_t, conv_transpose1d_u32)
#if defined(__HAVE_BFLOAT__)
CONVT1D_OP(bfloat, float, conv_transpose1d_bf16)
#endif

CONVT2D_OP(float, float, conv_transpose2d_f32)
CONVT2D_OP(half, float, conv_transpose2d_f16)
#if defined(__HAVE_BFLOAT__)
CONVT2D_OP(bfloat, float, conv_transpose2d_bf16)
#endif

CONV2D_GROUPED_DIRECT_OP(float, conv2d_grouped_direct_f32)

// Tile variants for the sweep. Threadgroup memory used is
// (CI_CHUNK * 3 * (TILE_T + 2) + CI_CHUNK * 288) * 4 bytes, against a 32 KB limit; thread count is
// (TILE_T / T_REG) * (32 / CO_REG), against a 1024 limit.
CONV2D_GROUPED_TILED_OP(float, 64, 8, 8, 4, conv2d_grouped_tiled_f32_t64_c8_r8x4)
CONV2D_GROUPED_TILED_OP(float, 64, 16, 8, 4, conv2d_grouped_tiled_f32_t64_c16_r8x4)
CONV2D_GROUPED_TILED_OP(float, 128, 8, 8, 4, conv2d_grouped_tiled_f32_t128_c8_r8x4)
CONV2D_GROUPED_TILED_OP(float, 128, 8, 4, 8, conv2d_grouped_tiled_f32_t128_c8_r4x8)
CONV2D_GROUPED_TILED_OP(float, 128, 8, 8, 8, conv2d_grouped_tiled_f32_t128_c8_r8x8)
CONV2D_GROUPED_TILED_OP(float, 256, 4, 8, 4, conv2d_grouped_tiled_f32_t256_c4_r8x4)
CONV2D_GROUPED_TILED_OP(float, 256, 4, 8, 8, conv2d_grouped_tiled_f32_t256_c4_r8x8)
