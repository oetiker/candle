#include <metal_stdlib>
using namespace metal;

// Utils
METAL_FUNC uint get_strided_index(
    uint idx,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides
) {
    // `dims` and `strides` are `size_t`, so `idx % dims[d]` promotes the whole expression to 64
    // bits and costs a software-emulated 64-bit division AND modulo per dimension, per element.
    // Narrowing to `uint` first is safe for the same reason the index itself is a `uint`: the
    // tensor has at most 2^32 elements. See the longer note in binary.metal -- six files carry
    // a private copy of this function and `utils.metal`'s narrowed version reaches none of them.
    const uint nd = uint(num_dims);
    uint strided_i = 0;
    for (uint d = 0; d < nd; d++) {
        uint dim_idx = nd - 1 - d;
        uint dim = uint(dims[dim_idx]);
        uint next = idx / dim;
        strided_i += (idx - next * dim) * uint(strides[dim_idx]);
        idx = next;
    }
    return strided_i;
}

template<uint Y>
constexpr uint div_ceil(uint x) {
    return x / Y + (x % Y > 0);
}

template<uint X, uint Y>
constexpr uint div_ceil() {
    return X / Y + (X % Y > 0);
}

template<typename T>
constexpr uint work_per_thread() {
    return div_ceil<8, sizeof(T)>();
}

// Kernels
template <typename T, int W = work_per_thread<T>()>
[[kernel]] void affine_kernel(
    constant size_t &dim,
    constant float &mul,
    constant float &add,
    device const T *input,
    device T *output,
    uint tid [[thread_position_in_grid]]
) {
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        output[i] = static_cast<T>(fma(float(input[i]), mul, add));
    }
}

template <typename T>
[[kernel]] void affine_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant float &mul,
    constant float &add,
    constant const T *input,
    device T *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) return;
    uint idx = get_strided_index(tid, num_dims, dims, strides);
    float result = fma(float(input[idx]), mul, add);
    output[tid] = static_cast<T>(result);
}

template <typename T, int W = work_per_thread<T>()>
[[kernel]] void powf_kernel(
    constant size_t &dim,
    constant float &mul,
    device const T *input,
    device T *output,
    uint tid [[thread_position_in_grid]]
) {
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        output[i] = static_cast<T>(pow(static_cast<float>(input[i]), mul));
    }
}

template <typename T>
[[kernel]] void powf_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant float &mul,
    constant const T *input,
    device T *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) return;
    uint idx = get_strided_index(tid, num_dims, dims, strides);
    output[tid] = static_cast<T>(pow(static_cast<float>(input[idx]), mul));
}

template <typename T, int W = work_per_thread<T>()>
[[kernel]] void elu_kernel(
    constant size_t &dim,
    constant float &mul,
    device const T *input,
    device T *output,
    uint tid [[thread_position_in_grid]]
) {
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        const T x = input[i];
        output[i] = static_cast<T>((x > 0) ? x : mul * (exp(x) - 1));
    }
}

template <typename T>
[[kernel]] void elu_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant float &mul,
    constant const T *input,
    device T *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) return;
    uint idx = get_strided_index(tid, num_dims, dims, strides);
    const T x = input[idx];
    output[tid] = static_cast<T>((x > 0) ? x : mul * (exp(x) - 1));
}

// Macros to help initialize kernels
#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#define init_affine(tname, t)                                           \
    init_kernel("affine_" #tname, affine_kernel, t)                     \
    init_kernel("affine_" #tname "_strided", affine_kernel_strided, t)

#define init_powf(tname, t)                                         \
    init_kernel("powf_" #tname, powf_kernel, t)                     \
    init_kernel("powf_" #tname "_strided", powf_kernel_strided, t)

#define init_elu(tname, t)                                          \
    init_kernel("elu_" #tname, elu_kernel, t)                       \
    init_kernel("elu_" #tname "_strided", elu_kernel_strided, t)


init_affine(u8, uint8_t);
init_affine(u32, uint32_t);
init_affine(i64, int64_t);
init_affine(f32, float);
init_affine(f16, half);

init_powf(f32, float);
init_powf(f16, half);

init_elu(f32, float);
init_elu(f16, half);

#if defined(__HAVE_BFLOAT__)
init_affine(bf16, bfloat);
init_powf(bf16, bfloat);
init_elu(bf16, bfloat);
#endif
