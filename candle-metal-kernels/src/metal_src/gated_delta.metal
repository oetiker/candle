// Fused GatedDeltaNet recurrence — the WHOLE `for t = 0..T` loop in ONE launch.
//
// Transliterated from mlx's `_make_gated_delta_kernel` (scalar-g, no-mask variant) in
// mlx_lm/models/gated_delta.py. The recurrent state lives in REGISTERS across every time step and
// never touches memory in between: that is the entire point, and it is why one launch covers
// prefill and decode alike.
//
// TWO DELIBERATE DIVERGENCES FROM MLX, both load-bearing:
//
//  1. NO in-kernel head mapping. mlx derives `hk_idx = hv_idx / (Hv / Hk)` — a GROUPED mapping.
//     Our checkpoint is a GGUF whose q/k broadcast is CYCLIC (`h % num_k_heads`, ggml's
//     `ggml_repeat_4d`), proven 84/84 against llama.cpp; at repeat_n == 2 the two mappings
//     DISAGREE and the wrong one emits fluent, plausible German. So q and k arrive here ALREADY
//     expanded to `Hk == Hv` by `repeat_tiled` on the Rust side, and this kernel's mapping is the
//     identity — it cannot disagree with anything. See `repeat_tiled_is_cyclic_not_grouped`.
//
//  2. STATE LAYOUT IS [B, H, Dk, Dv], not mlx's [B, H, Dv, Dk]. That is the layout the prompt
//     cache already snapshots (`recurrent_state`), and changing it is out of scope. Only the
//     initial load and the final store are affected — one strided access each, per launch.
//
// `g` is the DECAY ITSELF, exp'd by the caller, matching mlx's `compute_g` (which returns
// `exp(...)`). It is NOT the log-decay the chunked scan consumes.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Dk must be a multiple of 32: the state row is spread across one 32-wide SIMD group, n_per_t
// values per lane, and both reductions are `simd_sum` over exactly those 32 lanes.
template <typename InT, typename StT, int Dk, int Dv>
[[kernel]] void gated_delta(
    device const InT *q [[buffer(0)]],
    device const InT *k [[buffer(1)]],
    device const InT *v [[buffer(2)]],
    device const InT *g [[buffer(3)]],
    device const InT *beta [[buffer(4)]],
    device const StT *state_in [[buffer(5)]],
    device InT *y [[buffer(6)]],
    device StT *state_out [[buffer(7)]],
    constant const int &T [[buffer(8)]],
    constant const int &H [[buffer(9)]],
    uint3 tpig [[thread_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint tiis [[thread_index_in_simdgroup]]) {
  const int n = int(tpig.z); // flat (batch, head)
  const int b_idx = n / H;
  const int h_idx = n % H;
  constexpr int n_per_t = Dk / 32;

  // q, k: [B, T, H, Dk]  (H == Hk == Hv; see divergence 1 above)
  device const InT *q_ = q + (b_idx * T * H + h_idx) * Dk;
  device const InT *k_ = k + (b_idx * T * H + h_idx) * Dk;
  // v, y: [B, T, H, Dv]
  device const InT *v_ = v + (b_idx * T * H + h_idx) * Dv;
  device InT *y_ = y + (b_idx * T * H + h_idx) * Dv;

  const int dk_lane = int(tpitg.x); // 0..31, the SIMD lane
  const int dv_idx = int(tpig.y);   // this thread owns one row of the state

  // state_in, state_out: [B, H, Dk, Dv] — element (n, s_idx, dv_idx) is at
  // (n * Dk + s_idx) * Dv + dv_idx. Strided by Dv, unlike mlx's contiguous [.., Dv, Dk].
  device const StT *i_state = state_in + n * Dk * Dv;
  device StT *o_state = state_out + n * Dk * Dv;

  float state[n_per_t];
  for (int i = 0; i < n_per_t; ++i) {
    const int s_idx = n_per_t * dk_lane + i;
    state[i] = static_cast<float>(i_state[s_idx * Dv + dv_idx]);
  }

  // g, beta: [B, T, H]
  device const InT *g_ = g + b_idx * T * H;
  device const InT *beta_ = beta + b_idx * T * H;

  for (int t = 0; t < T; ++t) {
    const float gv = static_cast<float>(g_[h_idx]);
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      const int s_idx = n_per_t * dk_lane + i;
      state[i] = state[i] * gv;
      kv_mem += state[i] * static_cast<float>(k_[s_idx]);
    }
    kv_mem = simd_sum(kv_mem);

    const float delta =
        (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[h_idx]);

    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      const int s_idx = n_per_t * dk_lane + i;
      state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
      out += state[i] * static_cast<float>(q_[s_idx]);
    }
    out = simd_sum(out);
    if (tiis == 0) {
      y_[dv_idx] = static_cast<InT>(out);
    }

    // Advance to the next time step.
    q_ += H * Dk;
    k_ += H * Dk;
    v_ += H * Dv;
    y_ += H * Dv;
    g_ += H;
    beta_ += H;
  }

  for (int i = 0; i < n_per_t; ++i) {
    const int s_idx = n_per_t * dk_lane + i;
    o_state[s_idx * Dv + dv_idx] = static_cast<StT>(state[i]);
  }
}

// candle's Metal sources are compiled STATICALLY, so `Dk`/`Dv` cannot be templated at runtime the
// way mlx does it. Instantiate the shapes this checkpoint actually uses; the Rust side builds the
// mangled name and BAILS LOUDLY on anything else rather than falling back.
#define instantiate_gated_delta(tname, itype, stype, dk, dv)                    \
  template [[host_name("gated_delta_" #tname "_dk" #dk "_dv" #dv)]] [[kernel]]  \
  void gated_delta<itype, stype, dk, dv>(                                       \
      device const itype *q [[buffer(0)]],                                      \
      device const itype *k [[buffer(1)]],                                      \
      device const itype *v [[buffer(2)]],                                      \
      device const itype *g [[buffer(3)]],                                      \
      device const itype *beta [[buffer(4)]],                                   \
      device const stype *state_in [[buffer(5)]],                               \
      device itype *y [[buffer(6)]],                                            \
      device stype *state_out [[buffer(7)]],                                    \
      constant const int &T [[buffer(8)]],                                      \
      constant const int &H [[buffer(9)]],                                      \
      uint3 tpig [[thread_position_in_grid]],                                   \
      uint3 tpitg [[thread_position_in_threadgroup]],                           \
      uint tiis [[thread_index_in_simdgroup]]);

instantiate_gated_delta(f32, float, float, 128, 128)
