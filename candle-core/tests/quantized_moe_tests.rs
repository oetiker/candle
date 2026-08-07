//! Metal fused-MoE (`mul_mv_id`) and `QTensor::byte_view` tests.
//!
//! These live in candle-core rather than candle-metal-kernels because the thing under test is
//! `call_quantized_matmul_mv_id`, and only candle-core can *make* a quantized tensor
//! (`QTensor::quantize`) to feed it. Reaching the kernel through `QTensor::indexed_moe_forward`
//! exercises the whole argument-marshalling path, which is where the bugs are.
#![cfg(feature = "metal")]

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Result, Tensor};

/// Every dtype `call_quantized_matmul_mv_id` names, in its dispatch order, with whether it is
/// expected to work. The point of enumerating ALL of them is that a dtype which is merely *named*
/// in the match arm, but whose kernel wants a different grid or strides than the dispatch sends,
/// produces plausible garbage rather than an error -- f16/f32 did exactly that before they were
/// refused.
const DISPATCH_TABLE: &[(GgmlDType, bool)] = &[
    (GgmlDType::Q4K, true),
    (GgmlDType::Q5K, true),
    (GgmlDType::Q6K, true),
    (GgmlDType::Q8_0, true),
    (GgmlDType::Q4_0, true),
    (GgmlDType::Q4_1, true),
    (GgmlDType::Q5_0, true),
    (GgmlDType::Q5_1, true),
    (GgmlDType::Q3K, true),
    // Refused on purpose. Q2K's impl writes 8 rows per threadgroup while the shared threadgroup
    // table says 4, so its dispatch overruns into the next expert slot; f16/f32 route into
    // kernel_mul_mv_impl, which maps tgpig.x to ONE row and needs a real nb01; the rest have no
    // _id kernel in ggml at all.
    (GgmlDType::Q2K, false),
    (GgmlDType::F16, false),
    (GgmlDType::F32, false),
    (GgmlDType::BF16, false),
    (GgmlDType::Q8_1, false),
    (GgmlDType::Q8K, false),
];

const N_EXPERTS: usize = 4;
const N: usize = 64; // divisible by 8/4/2, the rows-per-threadgroup of every quantized mv kernel
const K: usize = 256; // a whole number of blocks for every dtype here
const BATCH: usize = 3;
const TOPK: usize = 2;

fn pseudo_random(n: usize, seed: f32) -> Vec<f32> {
    // Deterministic and dependency-free. Values spread over roughly [-1, 1] with no periodicity
    // that lines up with a 256-element block.
    (0..n)
        .map(|i| (i as f32 * 0.7351 + seed).sin() * 0.9 + (i as f32 * 0.13).cos() * 0.1)
        .collect()
}

/// `mul_mv_id` over a stacked `[n_experts, n, k]` weight must equal `n_experts` separate `mv_t`
/// calls against the same experts. Both go through the same `kernel_mul_mv_*_f32_impl`, so this
/// asserts EXACT equality: any difference means the fused path is reading the wrong bytes, not
/// that it rounds differently.
///
/// `ids` deliberately repeats an expert and leaves one unused, so a kernel that ignored the id
/// and walked experts in order would still be caught.
#[test]
fn mv_id_matches_per_expert_mv() -> Result<()> {
    let device = Device::new_metal(0)?;
    let ids_host: Vec<u32> = vec![3, 1, 0, 3, 1, 1];
    assert_eq!(ids_host.len(), BATCH * TOPK);

    // Every tensor below is built from its own host slice, never by narrowing a bigger one.
    // `Tensor::contiguous` is a NO-OP on a narrowed-but-contiguous view (the layout already
    // reports contiguous, it just carries a start offset), and `QTensor::quantize` reads the whole
    // underlying storage without applying the layout -- so `w.narrow(0, e, 1).contiguous()` would
    // quantize the ENTIRE stack for every expert and the reference would be silently wrong.
    let w_host = pseudo_random(N_EXPERTS * N * K, 1.0);
    let x_host = pseudo_random(BATCH * K, 2.0);
    let xk_host = pseudo_random(BATCH * TOPK * K, 3.0);

    for &(dtype, supported) in DISPATCH_TABLE {
        let w = Tensor::from_vec(w_host.clone(), (N_EXPERTS, N, K), &device)?;
        let stacked = match QTensor::quantize(&w, dtype) {
            Ok(q) => q,
            Err(e) => {
                assert!(
                    !supported,
                    "{dtype:?}: quantize failed but dtype is supported: {e}"
                );
                continue;
            }
        };

        let ids = Tensor::from_vec(ids_host.clone(), (BATCH, TOPK), &device)?;

        // in_dim1 == 1: one token vector broadcast to every slot (the gate/up shape).
        let x = Tensor::from_vec(x_host.clone(), (BATCH, 1, K), &device)?;
        let got = stacked.indexed_moe_forward(&x, &ids);

        if !supported {
            assert!(
                got.is_err(),
                "{dtype:?} is not dispatchable but indexed_moe_forward returned Ok -- a silently \
                 wrong result is exactly what refusing it is meant to prevent"
            );
            continue;
        }
        let got = got?.to_vec3::<f32>()?;

        // Reference: quantize each expert on its own. Blockwise quantization runs over the
        // flattened array and every dtype here has a block that divides K, so expert e's bytes
        // are the same whether it was quantized alone or as part of the stack.
        let mut experts = Vec::new();
        for e in 0..N_EXPERTS {
            let we =
                Tensor::from_vec(w_host[e * N * K..(e + 1) * N * K].to_vec(), (N, K), &device)?;
            experts.push(QMatMul::from_qtensor(QTensor::quantize(&we, dtype)?)?);
        }
        for b in 0..BATCH {
            let xb = Tensor::from_vec(x_host[b * K..(b + 1) * K].to_vec(), (1, K), &device)?;
            for j in 0..TOPK {
                let e = ids_host[b * TOPK + j] as usize;
                let want = experts[e].forward(&xb)?.reshape(N)?.to_vec1::<f32>()?;
                assert_eq!(
                    got[b][j], want,
                    "{dtype:?}: mv_id row (batch {b}, slot {j}, expert {e}) != mv_t"
                );
            }
        }

        // in_dim1 == topk: every slot has its own input row (the down-projection shape).
        let xk = Tensor::from_vec(xk_host.clone(), (BATCH, TOPK, K), &device)?;
        let got = stacked.indexed_moe_forward(&xk, &ids)?.to_vec3::<f32>()?;
        for b in 0..BATCH {
            for j in 0..TOPK {
                let e = ids_host[b * TOPK + j] as usize;
                let off = (b * TOPK + j) * K;
                let xbj = Tensor::from_vec(xk_host[off..off + K].to_vec(), (1, K), &device)?;
                let want = experts[e].forward(&xbj)?.reshape(N)?.to_vec1::<f32>()?;
                assert_eq!(
                    got[b][j], want,
                    "{dtype:?}: mv_id row (batch {b}, slot {j}, expert {e}) != mv_t, per-slot input"
                );
            }
        }
    }
    Ok(())
}

/// `byte_view` must reject an offset that is 256-byte aligned but lands mid-block.
///
/// Q6_K is the discriminator: its blocks are 210 bytes and gcd(210, 256) = 2, so 256-alignment
/// says nothing about block alignment. A mid-block view reinterprets the tail of one block as the
/// head of the next -- plausible numbers, no crash, and no other check would notice.
#[test]
fn byte_view_rejects_misblocked_offset() -> Result<()> {
    let device = Device::new_metal(0)?;
    let rows = 512;
    let host = pseudo_random(rows * K, 4.0);
    let t = QTensor::quantize(
        &Tensor::from_vec(host.clone(), (rows, K), &device)?,
        GgmlDType::Q6K,
    )?;
    let type_size = GgmlDType::Q6K.type_size();
    assert_eq!(
        type_size, 210,
        "this test's premise is Q6_K's 210-byte block"
    );

    // 256 is a multiple of 256 and not of 210, so it passes the Metal alignment and fails the
    // block alignment. If only the 256 check existed, this would be accepted.
    assert_eq!(256 % 256, 0);
    assert_ne!(256 % type_size, 0);
    let view_rows = 64;
    let err = t
        .byte_view(256, view_rows * K / 256 * type_size, (view_rows, K))
        .expect_err("a mid-block offset must be rejected, not silently aliased");
    let msg = err.to_string();
    assert!(
        msg.contains("whole number of") && msg.contains("Q6K"),
        "expected a block-alignment error, got: {msg}"
    );

    // A block-aligned, 256-aligned offset with a size that disagrees with the shape must also be
    // rejected: kernels size their reads from the SHAPE, so a short view is read past its end.
    let stride = view_rows * K / 256 * type_size;
    let aligned = 26880; // lcm(210, 256): the first offset satisfying both alignments
    assert_eq!(aligned % 256, 0);
    assert_eq!(aligned % type_size, 0);
    let err = t
        .byte_view(aligned, stride - type_size, (view_rows, K))
        .expect_err("a size that disagrees with the shape must be rejected");
    assert!(
        err.to_string().contains("do not describe"),
        "expected a size-vs-shape error, got: {err}"
    );

    // The honest case still works, so the test above is not passing for want of a valid input.
    let ok = t
        .byte_view(aligned, stride, (view_rows, K))?
        .expect("Metal must be able to alias a properly aligned view");
    assert_eq!(ok.shape().dims(), &[view_rows, K]);
    assert_eq!(ok.dtype(), GgmlDType::Q6K);
    assert_eq!(ok.storage_size_in_bytes(), stride);

    // And it reads the bytes it was pointed at, not the ones at offset 0. The reference is built
    // from the host rows the view should be looking at -- again never by narrowing, for the
    // `contiguous`/`quantize` reason above.
    let x = Tensor::from_vec(pseudo_random(K, 5.0), (1, K), &device)?;
    let via_view = QMatMul::from_qtensor(ok)?.forward(&x)?.to_vec2::<f32>()?;
    let start_row = aligned / type_size * GgmlDType::Q6K.block_size() / K;
    let want_rows = Tensor::from_vec(
        host[start_row * K..(start_row + view_rows) * K].to_vec(),
        (view_rows, K),
        &device,
    )?;
    let want = QMatMul::from_qtensor(QTensor::quantize(&want_rows, GgmlDType::Q6K)?)?
        .forward(&x)?
        .to_vec2::<f32>()?;
    assert_eq!(via_view, want, "a view must read from its own offset");
    assert_ne!(
        via_view,
        QMatMul::from_qtensor(QTensor::quantize(
            &Tensor::from_vec(host[..view_rows * K].to_vec(), (view_rows, K), &device)?,
            GgmlDType::Q6K
        )?)?
        .forward(&x)?
        .to_vec2::<f32>()?,
        "the offset must actually matter -- this view must NOT equal one at offset 0"
    );
    Ok(())
}
