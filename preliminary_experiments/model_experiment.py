import torch
from tqdm import tqdm
from pathlib import Path
import time
from utils import get_model, get_dataset, batch_masked_loss
from flipping import IntFlipper, FloatFlipper
from quantization import LNSQuantizer, Int8Quantizer, Float16Quantizer, Quantizer
from recogni.torch.ops.dtypes import LnsI4F3, FP8, LnsI5F10
from typing import List
import pandas as pd
from llms.model_cards import model_card_opt_125m


class Corruption:
    def __init__(self, quantizer: Quantizer, flip_probability: float):
        self.quantizer = quantizer
        self.name = quantizer.dtype

        if isinstance(quantizer, LNSQuantizer) or "int" in str(quantizer.dtype):
            self.flipper = IntFlipper(probability=flip_probability)
        elif "float" in str(quantizer.dtype):
            self.flipper = FloatFlipper(probability=flip_probability)

        self.invalid_entries_lst = []
        self.bit_error_mae_lst = []
        self.unperturbed_error_mae_lst = []
        self.perturbed_error_mae_lst = []

        self.invalid_output_entries_lst = []
        self.distance_output_lst = []
        self.num_different_predictions_lst = []

        self.mae = torch.nn.L1Loss()

    def get_mask(self, tensor: torch.Tensor, return_invalid_entries: bool = True):
        inf_mask = torch.isinf(tensor)
        nan_mask = torch.isnan(tensor)
        mask = inf_mask | nan_mask
        if return_invalid_entries:
            invalid_entries = mask.sum(dim=list(range(1, len(mask.shape)))).tolist()
            return mask, invalid_entries
        return mask

    def __call__(
        self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
        quantized = self.quantizer.quantize(input)
        dequantized = self.quantizer.dequantize(quantized)

        faulty_quantized = self.flipper(quantized)
        faulty_dequantized = self.quantizer.dequantize(faulty_quantized)

        mask, invalid_entries = self.get_mask(faulty_dequantized)
        self.invalid_entries_lst.extend(invalid_entries)

        self.bit_error_mae_lst.extend(
            batch_masked_loss(dequantized, faulty_dequantized, mask, self.mae)
        )
        self.unperturbed_error_mae_lst.extend(
            batch_masked_loss(input, dequantized, mask, self.mae)
        )
        self.perturbed_error_mae_lst.extend(
            batch_masked_loss(input, faulty_dequantized, mask, self.mae)
        )

        hook_idx = module._forward_hooks.keys()
        if len(hook_idx) > 0:
            hook = module._forward_hooks.pop(
                list(hook_idx)[0]
            )  # otherwise this hook will be called again
        else:
            hook = None
        faulty_output = module(faulty_dequantized)
        if hook:
            module._forward_hooks[0] = hook

        mask, invalid_out_entries = self.get_mask(faulty_output)
        out_loss = batch_masked_loss(output, faulty_output, mask, self.mae)
        self.invalid_output_entries_lst.extend(invalid_out_entries)
        self.distance_output_lst.extend(out_loss)

        pred = output.argmax(dim=-1)
        faulty_pred = faulty_output.argmax(dim=-1)
        self.num_different_predictions_lst.extend(
            (pred != faulty_pred).sum(axis=1).tolist()
        )


class TrackerModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, corruptions: List[Corruption]):
        super().__init__()
        self.model = model
        self.corruptions = corruptions
        self.set_hooks()

    def set_hooks(self):
        def post_func(module, inputs, outputs):
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            for corruption in self.corruptions:
                corruption(module, inputs, outputs)

        self.model.lm_head._forward_hooks.clear()
        self.model.lm_head._forward_pre_hooks.clear()
        self.model.lm_head.register_forward_hook(post_func)

    def remove_hooks(self):
        self.model.lm_head._forward_hooks.clear()
        self.model.lm_head._forward_pre_hooks.clear()

    def forward(self, **kwargs):
        return self.model(**kwargs)


def experiment(model: torch.nn.Module, corruptions: List[Corruption], save_dir: Path):
    device = "cuda:0"
    model = model.to(device)
    tracked_model = TrackerModule(model, corruptions)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            tracked_model(**batch)

    for corrupt in tracked_model.corruptions:
        lists = [lst for lst in dir(corrupt) if lst.endswith("_lst")]
        buf = {lst: getattr(corrupt, lst) for lst in lists}
        pd.DataFrame(buf).to_json(save_dir / f"{corrupt.name}.json")

    tracked_model.remove_hooks()


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    from main import OUTPUT_DIR

    model, tokenizer = get_model(model_card_opt_125m)

    dataset = get_dataset(tokenizer, split="train")
    dataloader = DataLoader(
        dataset.select_columns(["input_ids", "attention_mask"]),
        collate_fn=DataCollatorWithPadding(tokenizer),
        batch_size=16,
        shuffle=False,
    )
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    for flip_probability in [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]:
        corruptions = [
            Corruption(quantizer, flip_probability)
            for quantizer in [
                Int8Quantizer(),
                Float16Quantizer(),
                LNSQuantizer(dtype=LnsI4F3, eb=1),
                LNSQuantizer(dtype=FP8, eb=1),
                LNSQuantizer(dtype=LnsI5F10, eb=1),
            ]
        ]

        save_dir = OUTPUT_DIR / "model_experiment" / timestamp / str(flip_probability)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        experiment(model, corruptions=corruptions, save_dir=save_dir)
