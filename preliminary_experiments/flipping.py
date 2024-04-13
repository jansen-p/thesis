"""Module for bit flip perturbations."""

from abc import ABC, abstractmethod

import torch


class Flipper(ABC):
    """Base class for bit flip perturbations."""

    def __init__(self, probability: float):
        self.probability = probability

    @abstractmethod
    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        """Perturbation of the input."""
        pass

    def _flip_bits(self, value: torch.Tensor, bits: int):
        dtype = value.dtype
        mask = torch.bernoulli(
            torch.full((bits, *list(value.shape)), self.probability)
        ).to(device=value.device, dtype=dtype)
        for bit_position in range(bits - 1):
            value ^= mask[bit_position] * (1 << bit_position)
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
        buf = value.clone()

        value_int = buf.view(getattr(torch, f"int{bits}"))

        new_value_int = self._flip_bits(value_int, bits)

        # 31868 -> 0111 1100 0111 1100 -> reinterpret at float16 -> NaN
        # sign: 0 exponent: 11111 mantissa 0001111100
        # https://stackoverflow.com/questions/8341395/what-is-a-subnormal-floating-point-number/53203428#53203428

        value_modified = new_value_int.view(buf.dtype)
        return value_modified
