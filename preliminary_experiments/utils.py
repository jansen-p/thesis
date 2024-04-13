from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from racon.torch.flyte.reporting import ModelCard
import torch
from llms.constants import HF_TOKEN


def batch_masked_loss(
    x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, loss: torch.nn.Module
):
    assert (
        x.shape == y.shape == mask.shape
    ), f"Shapes of x, y, and mask must match, got {x.shape}, {y.shape}, {mask.shape}"
    losses = []
    for xi, yi, maski in zip(x, y, mask):
        xi = xi[~maski]
        yi = yi[~maski]
        losses.append(loss(xi, yi).item())
    return losses


def get_model(model_card: ModelCard):
    model = AutoModelForCausalLM.from_pretrained(
        model_card.name,
        torch_dtype=torch.float32,
        cache_dir=f"/tmp/huggingface-cache/{model_card.name}",
        token=HF_TOKEN,
        revision=None,
    )
    model.tie_weights()
    model.requires_grad_(False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_card.name, use_fast=False, token=HF_TOKEN
    )
    return model, tokenizer


def get_dataset(tokenizer, split="test"):
    dataset = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split=split)
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=512)
    )
    return dataset
