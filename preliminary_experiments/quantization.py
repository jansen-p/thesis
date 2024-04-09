"""Module to define quantizers."""

import torch
from abc import ABC, abstractmethod
import recogni.torch.math.functional as rF
from torch.ao.quantization.observer import MinMaxObserver


class Quantizer(ABC):
    """Base class to define a quantizer."""

    def __init__(self, dtype) -> None:
        self.dtype = dtype

    @abstractmethod
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize the input tensor."""
        pass

    @abstractmethod
    def dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """De-quantize the input tensor."""
        pass


class Int8Quantizer(Quantizer):
    """Int8 quantizer."""

    def __init__(self) -> None:
        super().__init__(torch.qint8)
        self.observer = MinMaxObserver(dtype=self.dtype)

    def observe(self, inputs: torch.Tensor) -> None:
        """Observe the input tensor."""
        for input in inputs:
            self.observer(input)
        self.qparams = self.observer.calculate_qparams()

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize the input tensor."""
        self.observe(x)
        quantized = torch.quantize_per_tensor(
            x, self.qparams[0].item(), self.qparams[1].item(), self.dtype
        )
        return quantized

    def dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """De-quantize the input tensor."""
        return torch.dequantize(x)


class Float16Quantizer(Quantizer):
    """Float16 quantizer."""

    def __init__(self) -> None:
        super().__init__(torch.float16)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize the input tensor."""
        return x.half()

    def dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """De-quantize the input tensor."""
        return x.float()


class LNSQuantizer(Quantizer):
    """LNS quantizer."""

    def __init__(self, dtype, eb: int) -> None:
        super().__init__(dtype)
        self.eb = eb

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize the input tensor."""
        return rF.float_to_quant(x, self.dtype(self.eb))

    def dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """De-quantize the input tensor."""
        return rF.quant_to_float(x, self.dtype(self.eb))
