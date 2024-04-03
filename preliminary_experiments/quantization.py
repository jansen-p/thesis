"""Module to define quantizers."""

import torch
from torch.ao.quantization.observer import MinMaxObserver


class Quantizer:
    """Base class to define a quantizer."""

    def __init__(self, dtype) -> None:
        self.dtype = dtype
        self.observer = MinMaxObserver(dtype=self.dtype)

    def observe(self, inputs: torch.Tensor) -> None:
        """Observe the input tensor."""
        for input in inputs:
            self.observer(input)
        self.qparams = self.observer.calculate_qparams()

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize the input tensor."""
        return torch.quantize_per_tensor(x, self.qparams[0].item(), self.qparams[1].item(), self.dtype)

    def dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """De-quantize the input tensor."""
        return torch.dequantize(x)


class Int8Quantizer(Quantizer):
    """Int8 quantizer."""

    def __init__(self) -> None:
        super().__init__(torch.qint8)
