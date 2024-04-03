"""Module for bit flip perturbations."""

import random
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

    def _flip_bits_single_pos(self, value, bits: int):
        for bit_position in range(bits):
            if random.random() < self.probability:
                value ^= 1 << bit_position
        return value

    def _flip_bits(self, value, bits: int):
        return value.apply_(lambda x: self._flip_bits_single_pos(x, bits))


class IntFlipper(Flipper):
    """Bit flipping for integers."""

    def __init__(self, probability: float = 0.1):
        super().__init__(probability)

    def __call__(self, value: torch.Tensor):
        """Perturbation of the input."""
        assert value.dtype in [torch.qint8, torch.int8, torch.int16]
        bits = 16 if value.dtype == torch.int16 else 8
        value = value.clone()
        return self._flip_bits(value, bits)


class FloatFlipper(Flipper):
    """Bit flipping for floats."""

    def __init__(self, probability: float = 0.1):
        super().__init__(probability)

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        """Perturbation of the input."""
        assert value.dtype in [torch.float16, torch.float32]
        bits = 32 if value.dtype == torch.float32 else 16

        int_fct = getattr(np, f"uint{bits}")
        float_fct = getattr(np, f"float{bits}")

        int_value = np.frombuffer(float_fct(value).tobytes(), dtype=int_fct)[0]
        new_int_value = self._flip_bits(int_value, bits)
        modified_value = np.frombuffer(int_fct(new_int_value).tobytes(), dtype=float_fct)[0]
        return modified_value
