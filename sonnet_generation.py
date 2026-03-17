'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model
from lora import apply_lora_to_model, count_parameters

from optimizer import AdamW

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """GPT-2 model for sonnet generation via autoregressive language modeling."""
  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    self.use_lora = getattr(args, 'use_lora', False)
    if self.use_lora:
      apply_lora_to_model(self.gpt, r=args.lora_r, alpha=args.lora_alpha)
    else:
      # default: fine-tune the full model
      for param in self.gpt.parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """Return logits per token (full sequence) for autoregressive sonnet training."""
    outputs = self.gpt(input_ids, attention_mask)
    sequence_output = outputs['last_hidden_state'] # [batch_size, seq_len, hidden_size]
    # detach embedding weights for LoRA to avoid CUBLAS errors on T4
    if self.use_lora:
      logits = sequence_output @ self.gpt.word_embedding.weight.detach().T
    else:
      logits = self.gpt.hidden_state_to_token(sequence_output) # [batch_size, seq_len, vocab_size]
    return logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=1.0, top_p=0.9, max_length=128, num_beams=1, length_penalty=0.6,
               repetition_penalty=1.2):
    """Generate sonnet. num_beams=1 uses top-p sampling; num_beams>1 uses beam search."""
    device = self.get_device()
    if num_beams > 1:
      return self._generate_beam_search(encoding, max_length, num_beams, length_penalty, repetition_penalty, device)
    return self._generate_sampling(encoding, temperature, top_p, max_length, device)

  def _generate_beam_search(self, encoding, max_length, num_beams, length_penalty, repetition_penalty, device):
    """Beam search with repetition penalty to reduce repeated phrases."""
    token_ids = encoding.to(device)
    batch_size, prompt_len = token_ids.shape
    vocab_size = self.gpt.word_embedding.num_embeddings
    eos_id = self.tokenizer.eos_token_id

    token_ids = token_ids.repeat(num_beams, 1)
    attention_mask = torch.ones_like(token_ids, dtype=torch.int64, device=device)
    beam_scores = torch.zeros(num_beams, device=device)
    beam_scores[1:] = float('-inf')
    finished = []

    for _ in range(max_length - prompt_len):
      logits = self.forward(token_ids, attention_mask)
      next_token_logits = logits[:, -1, :].float().clone()
      # Repetition penalty: penalize tokens already in the generated part (not prompt)
      if repetition_penalty != 1.0 and token_ids.shape[1] > prompt_len:
        for b in range(num_beams):
          gen_tokens = token_ids[b, prompt_len:].tolist()
          for t in gen_tokens:
            if next_token_logits[b, t] > 0:
              next_token_logits[b, t] /= repetition_penalty
            else:
              next_token_logits[b, t] *= repetition_penalty
      next_log_probs = F.log_softmax(next_token_logits, dim=-1)
      next_scores = beam_scores.unsqueeze(-1) + next_log_probs
      next_scores = next_scores.view(-1)
      top_scores, top_indices = torch.topk(next_scores, num_beams)
      beam_idx = top_indices // vocab_size
      token_idx = top_indices % vocab_size

      token_ids = torch.cat([token_ids[beam_idx], token_idx.unsqueeze(-1)], dim=1)
      attention_mask = torch.cat([attention_mask[beam_idx], torch.ones((num_beams, 1), dtype=torch.int64, device=device)], dim=1)
      beam_scores = top_scores

      eos_mask = (token_idx == eos_id)
      if eos_mask.any():
        seq_len = token_ids.shape[1]
        norm = seq_len ** length_penalty
        for i in torch.where(eos_mask)[0]:
          finished.append((token_ids[i].clone(), (beam_scores[i] / norm).item()))
        if len(finished) >= num_beams:
          break
        continuing = ~eos_mask
        if not continuing.any():
          break
        n = min(eos_mask.sum().item(), continuing.sum().item())
        worst = torch.topk(beam_scores[eos_mask], n, largest=False)[1]
        best = torch.topk(beam_scores[continuing], n, largest=True)[1]
        cont_idx = torch.where(continuing)[0][best]
        fin_idx = torch.where(eos_mask)[0][worst]
        token_ids[fin_idx] = token_ids[cont_idx].clone()
        attention_mask[fin_idx] = attention_mask[cont_idx].clone()
        beam_scores[fin_idx] = beam_scores[cont_idx].clone()

    if finished:
      best_seq, _ = max(finished, key=lambda x: x[1])
    else:
      seq_lens = (token_ids != eos_id).sum(dim=1).float() + 1
      norm = seq_lens ** length_penalty
      best_idx = (beam_scores / norm).argmax().item()
      best_seq = token_ids[best_idx]

    decoded = self.tokenizer.decode(best_seq.cpu().tolist())
    decoded = decoded.replace('<|endoftext|>', '').strip()
    return best_seq.unsqueeze(0), decoded

  def _generate_sampling(self, encoding, temperature, top_p, max_length, device):
    """Top-p (nucleus) sampling with temperature."""
    token_ids = encoding.to(device)
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(device)

    for _ in range(max_length):
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature
      probs = F.softmax(logits_last_token, dim=-1)
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
      top_p_mask[..., 0] = True
      filtered_probs = sorted_probs * top_p_mask
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.int64).to(device)], dim=1)

    decoded = self.tokenizer.decode(token_ids[0].cpu().tolist())
    decoded = decoded.replace('<|endoftext|>', '').strip()
    return token_ids, decoded


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train GPT-2 for sonnet generation on the Shakespeare sonnets dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  total, trainable = count_parameters(model)
  print(f"Total params: {total:,} | Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")

  lr = args.lr
  trainable_params = [p for p in model.parameters() if p.requires_grad]
  optimizer = AdamW(trainable_params, lr=lr)

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{batch[1]}{output[1]}\n\n')

    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  ckpt_path = getattr(args, 'ckpt', None) or f'{args.epochs-1}_{args.filepath}'
  saved = torch.load(ckpt_path, weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Generate for Sonnet Test
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)
  generated_sonnets = []
  for batch in tqdm(held_out_sonnet_dataset, desc='Test sonnets', disable=TQDM_DISABLE):
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(
      encoding['input_ids'],
      temperature=args.temperature, top_p=args.top_p,
      num_beams=getattr(args, 'num_beams', 1),
      length_penalty=getattr(args, 'length_penalty', 0.6),
      repetition_penalty=getattr(args, 'repetition_penalty', 1.2)
    )
    decoded_output = output[1]  # generate returns (token_ids, decoded_string)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])

  # Generate for Sonnet Dev (for leaderboard eval)
  held_out_dev = SonnetsDataset(args.held_out_sonnet_dev_path)
  generated_dev = []
  for batch in tqdm(held_out_dev, desc='Dev sonnets', disable=TQDM_DISABLE):
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(
      encoding['input_ids'],
      temperature=args.temperature, top_p=args.top_p,
      num_beams=getattr(args, 'num_beams', 1),
      length_penalty=getattr(args, 'length_penalty', 0.6),
      repetition_penalty=getattr(args, 'repetition_penalty', 1.2)
    )
    decoded_output = output[1]  # generate returns (token_ids, decoded_string)
    full_sonnet = f'{decoded_output}\n\n'
    generated_dev.append((sonnet_id, full_sonnet))

  with open(args.sonnet_dev_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_dev:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])

  print(f"Wrote {args.sonnet_out} and {args.sonnet_dev_out}")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")
  parser.add_argument("--held_out_sonnet_dev_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_dev_out", type=str, default="predictions/generated_sonnets_dev.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters (defaults = best chrF from tuning).
  parser.add_argument("--temperature", type=float, help="Sampling temperature (num_beams=1 only).", default=1.0)
  parser.add_argument("--top_p", type=float, help="Nucleus sampling threshold (num_beams=1 only).", default=0.9)
  parser.add_argument("--num_beams", type=int, default=1,
                      help="Beam width. 1=sampling (fast), >1=beam search.")
  parser.add_argument("--length_penalty", type=float, default=0.6,
                      help="Beam search length penalty.")
  parser.add_argument("--repetition_penalty", type=float, default=1.2,
                      help="Penalize repeated tokens in beam search.")

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  # LoRA arguments
  parser.add_argument("--use_lora", action='store_true', help="Use LoRA instead of full fine-tuning")
  parser.add_argument("--lora_r", type=int, default=4, help="LoRA rank (try 1, 2, 4)")
  parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA scaling factor")

  # Generate-only (skip training, load checkpoint and generate)
  parser.add_argument("--generate_only", action='store_true',
                      help="Skip training; load checkpoint and regenerate both dev and test files.")
  parser.add_argument("--ckpt", type=str, default=None,
                      help="Checkpoint path (required if --generate_only). e.g. 9_10-1e-05-sonnet.pt")

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  if args.use_lora:
    args.filepath = f'{args.epochs}-{args.lr}-lora-r{args.lora_r}-sonnet.pt'
  else:
    args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'
  seed_everything(args.seed)  # Fix the seed for reproducibility.

  if args.generate_only:
    if not args.ckpt:
      raise SystemExit("--ckpt is required when using --generate_only")
    generate_submission_sonnets(args)
  else:
    train(args)
    generate_submission_sonnets(args)
