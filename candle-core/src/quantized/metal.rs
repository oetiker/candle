use super::{GgmlDType, QStorage};
use crate::backend::BackendStorage;
use crate::{DType, Layout, MetalDevice, MetalStorage, Result, Shape, D};
use candle_metal_kernels::metal::Buffer;
use std::sync::Arc;

pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
    buffer: Arc<Buffer>,
    /// Byte offset of this tensor's data inside `buffer`, and its byte length. Non-zero offsets
    /// come from [`QMetalStorage::view`]: several `QTensor`s can then share one allocation, which
    /// is how the routed experts of a stacked `[n_experts, n, k]` MoE weight are exposed both as
    /// one stack (for the fused `mul_mv_id` kernel) and as `n_experts` 2-D matrices (for the
    /// grouped loop) without storing the ~20 GB of expert weights twice.
    offset: usize,
    size: usize,
}

impl QMetalStorage {
    pub fn zeros(device: &MetalDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size = elem_count * dtype.type_size() / dtype.block_size();
        let buffer = device
            .new_buffer_builder()
            .with_zeros(size)
            .with_label("qstorage_zeros")
            .build()?;
        Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
            offset: 0,
            size,
        })
    }

    /// A borrow of `size` bytes at `offset` inside this storage's buffer, as a storage in its own
    /// right. No copy and no new allocation: the returned value keeps the same `Arc<Buffer>`.
    ///
    /// `offset` must satisfy BOTH alignments, and they are independent:
    ///
    /// - 256 bytes, for Metal. Apple silicon only demands 4, but 256 is the documented worst case
    ///   for `setBuffer:offset:`, so requiring it cannot produce a silently misaligned read on
    ///   some other device.
    /// - one whole block of `dtype`, for correctness. Every kernel here casts the base pointer to
    ///   `block_qX *` and indexes in blocks, so an offset landing mid-block would reinterpret the
    ///   tail of one block as the head of the next -- plausible numbers, no crash. The two
    ///   alignments do not imply each other: Q6_K blocks are 210 bytes and gcd(210, 256) = 2, so
    ///   an offset can be a clean multiple of 256 and still cut a block in half.
    ///
    /// `size` must likewise be a whole number of blocks, so the view can describe a shape at all.
    pub fn view(&self, offset: usize, size: usize) -> Result<Self> {
        let type_size = self.dtype.type_size();
        if !offset.is_multiple_of(256) {
            crate::bail!("QMetalStorage::view offset {offset} is not 256-byte aligned")
        }
        if !offset.is_multiple_of(type_size) {
            crate::bail!(
                "QMetalStorage::view offset {offset} is not a whole number of {:?} blocks \
                 ({type_size} bytes)",
                self.dtype
            )
        }
        if !size.is_multiple_of(type_size) {
            crate::bail!(
                "QMetalStorage::view size {size} is not a whole number of {:?} blocks \
                 ({type_size} bytes)",
                self.dtype
            )
        }
        if offset + size > self.size {
            crate::bail!(
                "QMetalStorage::view {offset}+{size} out of bounds for {} bytes",
                self.size
            )
        }
        Ok(Self {
            dtype: self.dtype,
            device: self.device.clone(),
            buffer: self.buffer.clone(),
            offset: self.offset + offset,
            size,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Byte offset of this storage's data inside [`QMetalStorage::buffer`].
    pub fn buffer_offset(&self) -> usize {
        self.offset
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<MetalStorage> {
        use crate::quantized::k_quants::GgmlType;

        let buffer = self
            .device
            .new_buffer_builder()
            .with_size(self.size)
            .with_label("qstorage_dequantize_blit")
            .build()?;
        {
            let mut blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, self.offset, &buffer, 0, self.size);
        }
        self.device.flush_and_wait_current()?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => {
                let vec: Vec<f32> = read_to_vec(&buffer, block_len);
                f32::to_float(&vec, &mut out);
            }
            GgmlDType::F16 => {
                let vec: Vec<half::f16> = read_to_vec(&buffer, block_len);
                half::f16::to_float(&vec, &mut out);
            }
            GgmlDType::BF16 => {
                let vec: Vec<half::bf16> = read_to_vec(&buffer, block_len);
                half::bf16::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_0 => {
                let vec: Vec<crate::quantized::BlockQ4_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_1 => {
                let vec: Vec<crate::quantized::BlockQ4_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_0 => {
                let vec: Vec<crate::quantized::BlockQ5_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_1 => {
                let vec: Vec<crate::quantized::BlockQ5_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_0 => {
                let vec: Vec<crate::quantized::BlockQ8_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_1 => {
                let vec: Vec<crate::quantized::BlockQ8_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q2K => {
                let vec: Vec<crate::quantized::BlockQ2K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ2K::to_float(&vec, &mut out);
            }
            GgmlDType::Q3K => {
                let vec: Vec<crate::quantized::BlockQ3K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ3K::to_float(&vec, &mut out);
            }
            GgmlDType::Q4K => {
                let vec: Vec<crate::quantized::BlockQ4K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4K::to_float(&vec, &mut out);
            }
            GgmlDType::Q5K => {
                let vec: Vec<crate::quantized::BlockQ5K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5K::to_float(&vec, &mut out);
            }
            GgmlDType::Q6K => {
                let vec: Vec<crate::quantized::BlockQ6K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ6K::to_float(&vec, &mut out);
            }
            GgmlDType::Q8K => {
                let vec: Vec<crate::quantized::BlockQ8K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8K::to_float(&vec, &mut out);
            }
        }

        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&out)
            .with_label("qstorage_dequantized")
            .build()?;
        Ok(MetalStorage::new(
            buffer,
            self.device.clone(),
            elem_count,
            DType::F32,
        ))
    }

    pub fn quantize(&mut self, src: &MetalStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let data = qcpu_storage.data()?;
        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&data)
            .with_label("qstorage_quantized")
            .build()?;
        self.replace_buffer(buffer, data.len())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &MetalStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let data = qcpu_storage.data()?;
        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&data)
            .with_label("qstorage_quantize_imatrix")
            .build()?;
        self.replace_buffer(buffer, data.len())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&data)
            .with_label("qstorage_quantize_imatrix_onto")
            .build()?;
        self.replace_buffer(buffer, data.len())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&data)
            .with_label("qstorage_quantize_onto")
            .build()?;
        self.replace_buffer(buffer, data.len())
    }

    /// Swap in freshly quantized contents. Refuses to do so for a view: a view borrows someone
    /// else's allocation, so re-pointing it would silently orphan the tensor it was carved from.
    fn replace_buffer(&mut self, buffer: Arc<Buffer>, size: usize) -> Result<()> {
        if self.offset != 0 {
            crate::bail!(
                "cannot quantize into a QMetalStorage view (offset {})",
                self.offset
            )
        }
        self.buffer = buffer;
        self.size = size;
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.size
    }

    pub fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids: &MetalStorage,
        ids_l: &Layout,
    ) -> Result<MetalStorage> {
        use crate::MetalError;

        if ids.dtype() != DType::U32 {
            crate::bail!("quantized embedding expects u32 ids, got {:?}", ids.dtype())
        }
        if !ids_l.is_contiguous() {
            crate::bail!("quantized embedding requires contiguous ids")
        }
        if !hidden.is_multiple_of(self.dtype.block_size()) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                self.dtype.block_size()
            )
        }
        let expected_size = rows * hidden * self.dtype.type_size() / self.dtype.block_size();
        if self.storage_size_in_bytes() != expected_size {
            crate::bail!(
                "quantized tensor has {} bytes, expected {expected_size}",
                self.storage_size_in_bytes()
            )
        }
        if self.offset != 0 {
            // call_quantized_get_rows takes no src offset; rather than read from the wrong place,
            // say so.
            crate::bail!("quantized embedding is not implemented for a QMetalStorage view")
        }
        let ids_len = ids_l.shape().elem_count();
        let device = self.device.clone();
        let dst = device
            .new_buffer_builder()
            .with_size_for(ids_len * hidden, DType::F32)
            .with_label("qembedding")
            .build()?;
        let encoder = device.command_encoder()?;
        candle_metal_kernels::call_quantized_get_rows(
            device.device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            hidden,
            hidden * self.dtype.type_size() / self.dtype.block_size(),
            ids_len,
            &self.buffer,
            ids.buffer(),
            ids_l.start_offset() * DType::U32.size_in_bytes(),
            &dst,
        )
        .map_err(MetalError::from)?;
        Ok(MetalStorage::new(
            dst,
            device.clone(),
            ids_len * hidden,
            DType::F32,
        ))
    }

    fn fwd_mv(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();

        // We always use a single batch dimension and stack all the tensors in the batch on the
        // second dimension as the implementation in candle-metal-kernels doesn't handle batch
        // properly.
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            n => crate::bail!("Invalid rank {n} for quantized matmul metal"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device
            .new_buffer_builder()
            .with_size_for(dst_shape.elem_count(), DType::F32)
            .with_label("qmatmul")
            .build()?;
        let encoder = device.command_encoder()?;
        // In some cases it would be better to use the mm variant, though it has its drawbacks
        // around memory alignment.
        for batch_id in 0..m {
            candle_metal_kernels::call_quantized_matmul_mv_t(
                device.device(),
                &encoder,
                device.kernels(),
                self.dtype.into(),
                (1, 1, n, k),
                storage.buffer(),
                (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes(),
                &self.buffer,
                self.offset,
                batch_id * n * DType::F32.size_in_bytes(),
                &dst,
            )
            .map_err(MetalError::from)?;
        }
        let dst_storage =
            crate::MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let n = self_shape.dim(D::Minus2)?;
        let k = self_shape.dim(D::Minus1)?;
        let mut dst_shape = src_shape.dims().to_vec();

        if src_shape.rank() < self_shape.rank() {
            crate::bail!(
                "input rank ({}) must be >= weight rank ({})",
                src_shape.rank(),
                self_shape.rank()
            )
        }

        if src_shape.dim(D::Minus2)? == 1 {
            return self.fwd_mv(self_shape, storage, layout);
        }

        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device
            .new_buffer_builder()
            .with_size_for(dst_shape.elem_count(), DType::F32)
            .with_label("qmatmul")
            .build()?;
        let encoder = device.command_encoder()?;

        assert_eq!(storage.dtype(), DType::F32);

        if self_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", self_shape.rank())
        }
        let src0_l = crate::Layout::contiguous(
            [vec![1; 4 - self_shape.rank()], self_shape.dims().to_vec()].concat(),
        );
        let src0_stride = src0_l
            .stride()
            .iter()
            .map(|x| {
                (*x as f32 * (self.dtype.type_size() as f32 / self.dtype.block_size() as f32))
                    as usize
            })
            .collect::<Vec<_>>();

        if src_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", src_shape.rank())
        }
        let src1_l = crate::Layout::contiguous(
            [vec![1; 4 - src_shape.rank()], src_shape.dims().to_vec()].concat(),
        );

        candle_metal_kernels::call_quantized_matmul_mm_t(
            device.device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            src0_l.dims(),
            &src0_stride,
            &self.buffer,
            self.offset,
            src1_l.dims(),
            &src1_l
                .stride()
                .iter()
                .map(|x| x * DType::F32.size_in_bytes())
                .collect::<Vec<_>>(),
            storage.buffer(),
            src1_l.start_offset() * storage.dtype().size_in_bytes(),
            dst_shape.dims(),
            0,
            &dst,
        )
        .map_err(MetalError::from)?;

        let dst_storage =
            crate::MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    /// Fused MoE matrix-vector against a stacked `[n_experts, n, k]` weight.
    ///
    /// Same contract as `QCudaStorage::indexed_moe_forward`, so a model can call one method and
    /// get the right kernel on either backend:
    /// - `self`  `[n_experts, n, k]` quantized
    /// - `input` `[batch, in_dim1, k]` f32, `in_dim1` either 1 (broadcast the token to every
    ///   expert slot, as gate/up want) or `topk` (one row per slot, as down wants)
    /// - `ids`   `[batch, topk]` u32
    /// - returns `[batch, topk, n]` f32
    ///
    /// The quant type used is `self.dtype`, i.e. the type of THIS tensor. The 35B's
    /// `ffn_down_exps` is Q5_K on 37 layers and Q6_K on 3, so nothing here may be hoisted to a
    /// per-model constant.
    pub fn indexed_moe_forward(
        &self,
        self_shape: &Shape,
        input: &MetalStorage,
        input_l: &Layout,
        ids: &MetalStorage,
        ids_l: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        let (n_experts, n, k) = self_shape.dims3()?;
        if !input_l.is_contiguous() || !ids_l.is_contiguous() {
            crate::bail!("indexed_moe_forward requires contiguous input and ids")
        }
        if input.dtype() != DType::F32 {
            crate::bail!(
                "indexed_moe_forward expects f32 input, got {:?}",
                input.dtype()
            )
        }
        if ids.dtype() != DType::U32 {
            crate::bail!("indexed_moe_forward expects u32 ids, got {:?}", ids.dtype())
        }
        let (batch, in_dim1, in_k) = input_l.shape().dims3()?;
        let (ids_batch, topk) = ids_l.shape().dims2()?;
        if in_k != k {
            crate::bail!("indexed_moe_forward input k {in_k} != weight k {k}")
        }
        if ids_batch != batch {
            crate::bail!("indexed_moe_forward batch mismatch: input {batch}, ids {ids_batch}")
        }
        if in_dim1 != 1 && in_dim1 != topk {
            crate::bail!("indexed_moe_forward input dim1 {in_dim1} must be 1 or topk {topk}")
        }
        // row_stride_bytes: bytes between consecutive OUTPUT ROWS within one expert's matrix.
        // Only `mul_mm_id` (the prefill path below) reads this directly; `mul_mv_id` derives its
        // own row addressing from `expert_stride_bytes` alone. Computed before the multiply-by-n
        // so it is exact whenever k % block_size == 0, which every valid GGUF k-quant tensor is.
        let row_stride_bytes = k / self.dtype.block_size() * self.dtype.type_size();
        let expert_stride_bytes = row_stride_bytes * n;
        if expert_stride_bytes * n_experts != self.size {
            crate::bail!(
                "indexed_moe_forward: {n_experts} experts of {expert_stride_bytes} bytes do not \
                 fill this tensor's {} bytes",
                self.size
            )
        }

        let dst_shape = Shape::from((batch, topk, n));
        let device = input.device().clone();
        let dst = device
            .new_buffer_builder()
            .with_size_for(dst_shape.elem_count(), DType::F32)
            .with_label("qmoe_mv_id")
            .build()?;
        let encoder = device.command_encoder()?;

        let input_base = input_l.start_offset() * DType::F32.size_in_bytes();
        let ids_base = ids_l.start_offset() * DType::U32.size_in_bytes();

        if batch == 1 {
            // Decode: one token, one `mul_mv_id` dispatch, one threadgroup per output row.
            candle_metal_kernels::call_quantized_matmul_mv_id(
                device.device(),
                &encoder,
                device.kernels(),
                self.dtype.into(),
                (n_experts, n, k),
                (batch, in_dim1, topk),
                expert_stride_bytes,
                &self.buffer,
                self.offset,
                input.buffer(),
                input_base,
                ids.buffer(),
                ids_base,
                &dst,
                0,
            )
            .map_err(MetalError::from)?;
        } else {
            // Prefill: `mul_mm_id`'s tiled simdgroup-matrix kernel, chunked along the token axis
            // so `topk * chunk` never exceeds what Metal's threadgroup memory can hold for the
            // kernel's `rowids` scratch array. This is not a tuning choice: dispatching the whole
            // prompt in one call is a hard failure on real hardware (5304 tokens * topk=8 is ~20x
            // the ~2048-row budget on an M1 Max's 32768-byte maxThreadgroupMemoryLength).
            //
            // `mul_mm_id_dst_rows_max` (candle-metal-kernels) is the SINGLE SOURCE OF TRUTH for
            // this ceiling -- `call_quantized_matmul_mm_id` calls the exact same function for its
            // own hard-reject check below. Do not re-derive this formula here: a second, drifted
            // copy would let this site size a chunk the callee then rejects.
            let dst_rows_max = candle_metal_kernels::mul_mm_id_dst_rows_max(device.device());
            // Chunk size is capped at 32, NOT maximized to `dst_rows_max`, for a reason that is
            // about correctness-in-practice rather than tuning: `kernel_mul_mm_id`'s row-id
            // collection ("// TODO: parallelize this loop" in quantized.metal, ggml's own words)
            // runs UNPARALLELIZED on every one of a threadgroup's 128 threads, scanning all
            // `chunk * topk` ids -- and every one of `n_experts` dispatched threadgroups per tile
            // pays that scan even when it finds no rows for its expert. That cost is
            // O(chunk^2 * n_experts): linear in chunk from the scan length, another linear factor
            // from grid width (`divide(chunk, 32)`) scaling with chunk too, AND linear in
            // `n_experts` directly since every expert's threadgroup pays the scan regardless of
            // whether it collects any rows.
            //
            // 32 is an EMPIRICAL FLOOR, not a derived safe value, and it is bounded ONLY for the
            // exact configuration it was measured on: 256 experts, topk=8, on an M1 Max, with a
            // 5304-token prompt. At that configuration: chunk = dst_rows_max/topk = 256 measured
            // indistinguishable from a GPU hang (killed after 20+ min of no progress); chunk = 64
            // measured 77s; chunk = 32 measured a real 68s for the whole prefill. Since the scan
            // cost scales with `n_experts` independently of `chunk`, a checkpoint with
            // MEANINGFULLY MORE EXPERTS OR A LARGER TOPK than this one could reintroduce
            // hang-like wall-clock even at chunk=32 -- this constant has not been re-validated
            // against any other configuration. Do not retune it (Task 10's concern); if it needs
            // to change, re-measure at the new configuration rather than assuming the old
            // measurement still holds.
            let max_batch_per_dispatch = (dst_rows_max / topk).clamp(1, 32);

            let in_row_bytes = in_dim1 * k * DType::F32.size_in_bytes();
            let ids_row_bytes = topk * DType::U32.size_in_bytes();
            let dst_row_bytes = topk * n * DType::F32.size_in_bytes();

            let mut done = 0usize;
            while done < batch {
                let chunk = (batch - done).min(max_batch_per_dispatch);
                candle_metal_kernels::call_quantized_matmul_mm_id(
                    device.device(),
                    &encoder,
                    device.kernels(),
                    self.dtype.into(),
                    (n_experts, n, k),
                    (chunk, in_dim1, topk),
                    row_stride_bytes,
                    expert_stride_bytes,
                    &self.buffer,
                    self.offset,
                    input.buffer(),
                    input_base + done * in_row_bytes,
                    ids.buffer(),
                    ids_base + done * ids_row_bytes,
                    &dst,
                    done * dst_row_bytes,
                )
                .map_err(MetalError::from)?;
                done += chunk;
            }
        }

        let dst_storage =
            crate::MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let buffer = self
            .device
            .new_buffer_builder()
            .with_size(self.size)
            .with_label("qstorage_data_blit")
            .build()?;
        {
            let mut blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, self.offset, &buffer, 0, self.size);
        }
        self.device.flush_and_wait_current()?;
        Ok(read_to_vec::<u8>(&buffer, self.storage_size_in_bytes()))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &MetalDevice,
    data: &[T],
) -> Result<QStorage> {
    let buffer = device
        .new_buffer_builder()
        .with_data(data)
        .with_label("qstorage_load_quantized")
        .build()?;
    let device = device.clone();
    let size = std::mem::size_of_val(data);
    Ok(QStorage::Metal(QMetalStorage {
        dtype: T::DTYPE,
        device,
        buffer,
        offset: 0,
        size,
    }))
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

impl From<GgmlDType> for candle_metal_kernels::GgmlDType {
    fn from(value: GgmlDType) -> Self {
        match value {
            GgmlDType::Q4_0 => candle_metal_kernels::GgmlDType::Q4_0,
            GgmlDType::Q4_1 => candle_metal_kernels::GgmlDType::Q4_1,
            GgmlDType::Q5_0 => candle_metal_kernels::GgmlDType::Q5_0,
            GgmlDType::Q5_1 => candle_metal_kernels::GgmlDType::Q5_1,
            GgmlDType::Q8_0 => candle_metal_kernels::GgmlDType::Q8_0,
            GgmlDType::Q8_1 => candle_metal_kernels::GgmlDType::Q8_1,
            GgmlDType::Q2K => candle_metal_kernels::GgmlDType::Q2K,
            GgmlDType::Q3K => candle_metal_kernels::GgmlDType::Q3K,
            GgmlDType::Q4K => candle_metal_kernels::GgmlDType::Q4K,
            GgmlDType::Q5K => candle_metal_kernels::GgmlDType::Q5K,
            GgmlDType::Q6K => candle_metal_kernels::GgmlDType::Q6K,
            GgmlDType::Q8K => candle_metal_kernels::GgmlDType::Q8K,
            GgmlDType::F16 => candle_metal_kernels::GgmlDType::F16,
            GgmlDType::F32 => candle_metal_kernels::GgmlDType::F32,
            GgmlDType::BF16 => candle_metal_kernels::GgmlDType::BF16,
        }
    }
}
