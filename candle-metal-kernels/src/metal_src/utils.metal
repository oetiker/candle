#pragma once
#include <metal_stdlib>
using namespace metal;

METAL_FUNC uint nonzero(uint n) {
    return n == 0 ? 1 : n;
}

template<uint N>
constexpr uint nonzero() {
    return N == 0 ? 1 : N;
}

template<typename T>
constexpr ushort granularity() {
    return nonzero<vec_elements<T>::value>();
}

METAL_FUNC uint next_p2(uint x) {
    return 1 << (32 - clz(x - 1));
}

METAL_FUNC uint prev_p2(uint x) {
    return 1 << (31 - clz(x));
}

constant uint MAX_SHARED_MEM = 32767;

template<typename T>
METAL_FUNC uint max_shared_mem(uint n) {
    return min(n, prev_p2(MAX_SHARED_MEM / sizeof(T)));
}

METAL_FUNC uint get_strided_index(
    uint idx,
    constant const uint &num_dims,
    constant const size_t *dims,
    constant const size_t *strides
) {
    // `dims` and `strides` are `size_t`, so `idx % dims[d]` promotes the whole expression to 64
    // bits and costs a software-emulated 64-bit division AND modulo per dimension, per element.
    // Narrowing them to `uint` first is safe for the same reason the index itself is a `uint`: the
    // tensor has at most 2^32 elements, so neither a coordinate into it nor a stride within it can
    // exceed that. This function is the inner loop of every strided unary, binary, ternary, cast,
    // affine, reduce and indexing kernel.
    uint strided_i = 0;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        uint dim = uint(dims[dim_idx]);
        uint next = idx / dim;
        strided_i += (idx - next * dim) * uint(strides[dim_idx]);
        idx = next;
    }
    return strided_i;
}
