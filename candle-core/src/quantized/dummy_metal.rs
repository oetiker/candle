#![allow(unused)]
use super::GgmlDType;
use crate::{Error, MetalDevice, MetalStorage, Result};

pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
}

impl QMetalStorage {
    pub fn zeros(_: &MetalDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<MetalStorage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    // Present only so `QTensor::byte_view` COMPILES without the metal feature -- its Metal arm
    // calls this, and the whole file is the not-compiled-with-metal stub, so no caller can ever
    // hold a real one to reach it. Without this method candle-core does not build at all on the
    // CPU-only configuration, which is what non-Apple CI and a CPU bit-identity gate need.
    pub fn view(&self, _offset: usize, _size: usize) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn quantize(&mut self, _src: &MetalStorage) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn quantize_imatrix(
        &mut self,
        _src: &MetalStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &crate::CpuStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn quantize_onto(&mut self, _src: &crate::CpuStorage) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn embedding(
        &self,
        _rows: usize,
        _hidden: usize,
        _ids: &MetalStorage,
        _ids_l: &crate::Layout,
    ) -> Result<MetalStorage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn fwd(
        &self,
        _self_shape: &crate::Shape,
        _storage: &MetalStorage,
        _layout: &crate::Layout,
    ) -> Result<(MetalStorage, crate::Shape)> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn indexed_moe_forward(
        &self,
        _: &crate::Shape,
        _: &MetalStorage,
        _: &crate::Layout,
        _: &MetalStorage,
        _: &crate::Layout,
    ) -> Result<(MetalStorage, crate::Shape)> {
        Err(Error::NotCompiledWithMetalSupport)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &MetalDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithMetalSupport)
}
