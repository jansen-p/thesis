import torch

from flipping import IntFlipper
from quantization import Int8Quantizer

if __name__ == "__main__":
    activations = torch.distributions.pareto.Pareto(0.1, 2).sample((512, 1024))

    iq = Int8Quantizer()
    iq.observe(activations)

    print(activations.shape, activations.mean(), activations.min(), activations.max())
    print(activations[:3, :3])
    quantized = iq.quantize(activations)
    print(quantized[:3, :3].int_repr())
    dequantized = iq.dequantize(quantized)
    print(dequantized[:3, :3])

    quantized = iq.quantize(activations)
    print(quantized[:3, :4])

    ifl = IntFlipper(probability=0.5)
    quantized_ints = ifl(quantized.int_repr())
    dequantized = torch._make_per_tensor_quantized_tensor(
        quantized_ints, quantized.q_scale(), quantized.q_zero_point()
    )
    print(dequantized[:3, :4])
