import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
  """LoRA wrapper around a frozen linear layer (Hu et al., 2021).
  Adds trainable low-rank matrices A and B on top of frozen pretrained weights.
  B is initialized to zeros so at the start the output is the same as the original layer.
  """

  def __init__(self, original_layer, r=4, lora_alpha=1.0):
    super().__init__()
    self.r = r
    self.alpha = lora_alpha

    # store frozen weights directly instead of as a submodule
    self.weight = original_layer.weight
    self.bias = original_layer.bias

    # A is (r x in_features), B is (out_features x r)
    self.A = nn.Parameter(torch.randn(r, original_layer.in_features) * 1e-2)
    self.B = nn.Parameter(torch.zeros(original_layer.out_features, r))

  def forward(self, x):
    base = F.linear(x, self.weight, self.bias)
    # go through the low-rank bottleneck instead of forming the full (out x in) matrix
    # which causes CUBLAS errors on T4 GPUs
    lora_update = (x @ self.A.T @ self.B.T) * (self.alpha / self.r)
    return base + lora_update


def apply_lora_to_model(gpt_model, r=4, alpha=1.0):
  """Freeze all weights then inject LoRA into Q and V projections
  in each attention layer (following Hu et al. 2021)."""
  for param in gpt_model.parameters():
    param.requires_grad = False

  for layer in gpt_model.gpt_layers:
    layer.self_attention.query = LoRALinear(layer.self_attention.query, r=r, lora_alpha=alpha)
    layer.self_attention.value = LoRALinear(layer.self_attention.value, r=r, lora_alpha=alpha)

  return gpt_model


def count_parameters(model):
  """Helper to count total and trainable parameters."""
  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return total, trainable
