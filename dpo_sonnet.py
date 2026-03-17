"""
Train SonnetGPT with Direct Preference Optimization (DPO).

Like dpo_paraphrase.py, this script trains with DPO then generates sonnets.
Pairs: (prompt, y_w, y_l) where y_w=gold completion, y_l=another sonnet's completion.

Run: python dpo_sonnet.py --use_gpu
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sonnet_generation import (
    SonnetGPT,
    add_arguments,
    seed_everything,
    generate_submission_sonnets,
)
from datasets import SonnetDPODataset
from optimizer import AdamW
from dpo import dpo_loss_sonnet

TQDM_DISABLE = False


def sequence_log_probs(logits, input_ids, attention_mask, prompt_lens):
    """
    Compute log p(completion|prompt) for each example in batch.
    logits: [B, T, V], input_ids: [B, T], attention_mask: [B, T]
    prompt_lens: list of B ints - index of last prompt token (completion starts at prompt_len+1)
    Returns: [B] tensor
    """
    B, T, V = logits.shape
    device = logits.device
    log_probs_list = []
    for i in range(B):
        plen = prompt_lens[i]
        seq_len = int(attention_mask[i].sum().item())
        if seq_len <= plen + 1:
            log_probs_list.append(torch.tensor(0.0, device=device))
            continue
        # Sum log P(token_k | ...) for k = plen+1 .. seq_len-1
        # logits at k-1 predicts token at k
        idx = torch.arange(plen + 1, seq_len, device=device)
        log_p = F.log_softmax(logits[i, idx - 1, :], dim=-1)
        token_ids = input_ids[i, idx]
        lp = log_p.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1).sum()
        log_probs_list.append(lp)
    return torch.stack(log_probs_list)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1, help="DPO runs 4 forwards per batch; use 1 if OOM")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--ckpt_base", type=str, default=None,
                        help="Load policy and ref from SFT checkpoint (recommended for best results)")
    parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
    parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
    parser.add_argument("--held_out_sonnet_dev_path", type=str, default="data/sonnets_held_out_dev.txt")
    parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets_dpo.txt")
    parser.add_argument("--sonnet_dev_out", type=str, default="predictions/generated_sonnets_dev_dpo.txt")
    parser.add_argument("--model_size", type=str, default="gpt2",
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--generate_only", action='store_true',
                        help="Skip training; load checkpoint and generate only.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path (required if --generate_only).")
    args = parser.parse_args()
    return args


def train_dpo_sonnet(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    args = add_arguments(args)

    train_dataset = SonnetDPODataset(args.sonnet_path)
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn
    )

    model = SonnetGPT(args)
    ref_model = SonnetGPT(args)
    if args.ckpt_base:
        ckpt = torch.load(args.ckpt_base, weights_only=False)
        model.load_state_dict(ckpt['model'])
        ref_model.load_state_dict(ckpt['model'])
        print(f"Loaded from {args.ckpt_base}")
    model = model.to(device)
    ref_model = ref_model.to(device)
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.
    )

    for epoch in range(args.epochs):
        model.train()
        ref_model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f'dpo-sonnet-{epoch}', disable=TQDM_DISABLE):
            ids_w = batch['input_ids_w'].to(device)
            mask_w = batch['attention_mask_w'].to(device)
            ids_l = batch['input_ids_l'].to(device)
            mask_l = batch['attention_mask_l'].to(device)
            prompt_lens = batch['prompt_len']

            optimizer.zero_grad()
            logits_theta_w = model(ids_w, mask_w)
            logits_theta_l = model(ids_l, mask_l)
            with torch.no_grad():
                logits_ref_w = ref_model(ids_w, mask_w)
                logits_ref_l = ref_model(ids_l, mask_l)

            log_theta_w = sequence_log_probs(logits_theta_w, ids_w, mask_w, prompt_lens)
            log_theta_l = sequence_log_probs(logits_theta_l, ids_l, mask_l, prompt_lens)
            log_ref_w = sequence_log_probs(logits_ref_w, ids_w, mask_w, prompt_lens)
            log_ref_l = sequence_log_probs(logits_ref_l, ids_l, mask_l, prompt_lens)

            loss = dpo_loss_sonnet(log_theta_w, log_theta_l, log_ref_w, log_ref_l, beta=args.beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch}: DPO loss = {total_loss / n_batches:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/dpo-{args.epochs}-b{args.beta}-sonnet.pt"
    torch.save({'model': model.state_dict(), 'args': args}, save_path)
    print(f"Saved model to {save_path}")
    return save_path


if __name__ == "__main__":
    args = get_args()
    seed_everything(11711)

    if args.generate_only:
        if not args.ckpt:
            raise SystemExit("--ckpt is required when using --generate_only")
        generate_submission_sonnets(args)
    else:
        save_path = train_dpo_sonnet(args)
        args.ckpt = save_path
        generate_submission_sonnets(args)
