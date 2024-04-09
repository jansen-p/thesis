import torch

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path("output")


def plot_distribution(distrib):
    activations = distrib.sample((10000,))
    plt.hist(activations.flatten(), bins=1000)
    plt.xlim(right=2)
    plt.ylim(top=1500)
    plt.savefig("distribution.png")


def experiment(
    flipper, quantizer, distrib, num_samples, save_dir: Path, sample_shape=(512, 1024)
):
    quantizer_name = quantizer.__class__.__name__
    if isinstance(quantizer.dtype, torch.dtype):
        quantizer_dtype = str(quantizer.dtype)  # torch.float16
    else:
        quantizer_dtype = quantizer.dtype.__name__

    out_dir = save_dir / f"{quantizer_name}_{quantizer_dtype}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(quantizer.__class__.__name__, quantizer.dtype)

    mae = torch.nn.L1Loss()

    unperturbed_error_lst = []
    perturbed_error_lst = []
    bit_error_lst = []

    unperturbed_error_mae_lst = []
    perturbed_error_mae_lst = []
    bit_error_mae_lst = []

    for _ in tqdm(range(num_samples)):
        activations = distrib.sample(sample_shape)

        quantized = quantizer.quantize(activations)
        dequantized = quantizer.dequantize(quantized)

        faulty_quantized = flipper(quantized)
        faulty_dequantized = quantizer.dequantize(faulty_quantized)

        bit_error_lst.append(torch.norm(dequantized - faulty_dequantized, "fro").item())
        unperturbed_error_lst.append(
            torch.norm(activations - dequantized, "fro").item()
        )
        perturbed_error_lst.append(
            torch.norm(activations - faulty_dequantized, "fro").item()
        )

        bit_error_mae_lst.append(mae(dequantized, faulty_dequantized).item())
        unperturbed_error_mae_lst.append(mae(activations, dequantized).item())
        perturbed_error_mae_lst.append(mae(activations, faulty_dequantized).item())

    results = pd.DataFrame(
        {
            "quantization_fro_error": unperturbed_error_lst,
            "quantization_flip_fro_error": perturbed_error_lst,
            "diff_fro": bit_error_lst,
            "quantization_mae_error": unperturbed_error_mae_lst,
            "quantization_flip_mae_error": perturbed_error_mae_lst,
            "diff_mae": bit_error_mae_lst,
        }
    )
    # results["error_ratio"] = results["perturbed"] / results["unperturbed"] - 1

    # plt.plot(results.keys(), list(map(lambda x: x["perturbed"]/x["unperturbed"]-1, results.values())))
    # plt.savefig(out_dir / "error_ratio.png")

    results.to_json(out_dir / "results.json")
