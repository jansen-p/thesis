"""Module for bit flip perturbations."""

from abc import ABC, abstractmethod

import numpy as np
import torch


class Flipper(ABC):
    """Base class for bit flip perturbations."""

    def __init__(self, probability: float):
        self.probability = probability

    @abstractmethod
    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        """Perturbation of the input."""
        pass

    def _flip_bits(self, value, bits: int):
        dtype = value.dtype
        for bit_position in range(bits):
            mask = np.random.binomial(n=1, p=self.probability, size=value.shape)
            value ^= mask * (1 << bit_position)
        return value.to(dtype=dtype)


class IntFlipper(Flipper):
    """Bit flipping for integers."""

    def __init__(self, probability: float = 0.1):
        super().__init__(probability)
        self.bits_map = {torch.qint8: 8, torch.int16: 16, torch.int8: 8, torch.uint8: 8}

    def __call__(self, value: torch.Tensor):
        """Perturbation of the input."""
        assert value.dtype in [torch.qint8, torch.int16, torch.int8, torch.uint8]

        value = value.clone()
        value_int = value.int_repr() if value.dtype == torch.qint8 else value
        value_int_flipped = self._flip_bits(value_int, bits=self.bits_map[value.dtype])
        if value.dtype == torch.qint8:
            quantized_value_int_flipped = torch._make_per_tensor_quantized_tensor(
                value_int_flipped, value.q_scale(), value.q_zero_point()
            )
            return quantized_value_int_flipped
        else:  # no conversion to quantized tensor needed
            return value_int_flipped


class FloatFlipper(Flipper):
    """Bit flipping for floats."""

    def __init__(self, probability: float = 0.1):
        super().__init__(probability)

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        """Perturbation of the input."""
        assert value.dtype in [torch.float16, torch.float32]
        bits = 32 if value.dtype == torch.float32 else 16

        value_int = value.view(getattr(torch, f"int{bits}"))

        new_value_int = self._flip_bits(value_int, bits)

        value_modified = new_value_int.view(value.dtype)
        return value_modified
