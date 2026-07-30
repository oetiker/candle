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
template <
    typename T,
    typename U,
    typename IR = T,
    int W = work_per_thread<T>()
>
[[kernel]] void cast_kernel(
    constant size_t &dim,
    device const T* input,
    device U* output,
    uint tid [[thread_position_in_grid]]
) {
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        output[i] = static_cast<U>(static_cast<IR>(input[i]));
    }
}

template <typename T, typename U, typename IR = T>
[[kernel]] void cast_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant const T *input,
    device U *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) return;
    output[tid] = static_cast<U>(
        static_cast<IR>(input[get_strided_index(tid, num_dims, dims, strides)])
    );
}

// Macros to help initialize kernels
#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#define init_cast(tname, t, uname, u)                                           \
    init_kernel("cast_" #tname "_" #uname, cast_kernel, t, u)                   \
    init_kernel("cast_" #tname "_" #uname "_strided", cast_kernel_strided, t, u)

#if defined(__HAVE_BFLOAT__)
#define init_cast_all(tname, t)         \
    init_cast(tname, t, f32, float)     \
    init_cast(tname, t, f16, half)      \
    init_cast(tname, t, bf16, bfloat)   \
    init_cast(tname, t, i64, int64_t)   \
    init_cast(tname, t, u32, uint32_t)  \
    init_cast(tname, t, u8, uint8_t)
#else
#define init_cast_all(tname, t)         \
    init_cast(tname, t, f32, float)     \
    init_cast(tname, t, f16, half)      \
    init_cast(tname, t, i64, int64_t)   \
    init_cast(tname, t, u32, uint32_t)  \
    init_cast(tname, t, u8, uint8_t)
#endif


init_cast_all(f32, float);
init_cast_all(f16, half);
#if defined(__HAVE_BFLOAT__)
init_cast_all(bf16, bfloat);
#endif
init_cast_all(i64, int64_t);
init_cast_all(u32, uint32_t);
init_cast_all(u8, uint8_t);
