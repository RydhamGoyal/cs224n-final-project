"""
Train ParaphraseGPT with Direct Preference Optimization (DPO).

Like paraphrase_detection.py (and LoRA), this script trains then tests:
  - Training: DPO loss on Quora paraphrase data
  - Testing: Eval on dev/test, writes predictions for submission

Run: python dpo_paraphrase.py --use_gpu
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from paraphrase_detection import ParaphraseGPT, add_arguments, seed_everything
from datasets import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    load_paraphrase_data,
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from optimizer import AdamW
from dpo import dpo_loss_paraphrase

TQDM_DISABLE = False


def get_dpo_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--ref_from", type=str, default=None,
                        help="Path to reference checkpoint. If None, uses untrained GPT-2.")
    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output-dpo.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output-dpo.csv")
    parser.add_argument("--model_size", type=str, default="gpt2",
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
    parser.add_argument("--test_only", action='store_true',
                        help="Skip training; load checkpoint and run evaluation only.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path (required if --test_only).")
    args = parser.parse_args()
    return args


def train_dpo(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    args = add_arguments(args)

    # Load data
    train_data = load_paraphrase_data(args.para_train)
    train_dataset = ParaphraseDetectionDataset(train_data, args)
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn
    )

    # Policy model (trainable)
    model = ParaphraseGPT(args)
    model = model.to(device)

    # Reference model (frozen copy)
    ref_model = ParaphraseGPT(args)
    if args.ref_from:
        ckpt = torch.load(args.ref_from, weights_only=False)
        ref_model.load_state_dict(ckpt['model'])
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

        for batch in tqdm(train_loader, desc=f'dpo-train-{epoch}', disable=TQDM_DISABLE):
            b_ids = batch['token_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Pairwise data: y_w = gold, y_l = opposite
            y_w = labels
            y_l = 1 - labels

            optimizer.zero_grad()
            logits_theta = model(b_ids, b_mask)
            with torch.no_grad():
                logits_ref = ref_model(b_ids, b_mask)

            loss = dpo_loss_paraphrase(logits_theta, logits_ref, y_w, y_l, beta=args.beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch}: DPO loss = {total_loss / n_batches:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/dpo-{args.epochs}-b{args.beta}-paraphrase.pt"
    torch.save({'model': model.state_dict(), 'args': args}, save_path)
    print(f"Saved model to {save_path}")
    return save_path


@torch.no_grad()
def test_dpo(args, ckpt_path):
    """Evaluate DPO model on dev and test; write predictions for submission."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(ckpt_path, weights_only=False)
    model_args = saved['args']
    model_args.use_gpu = args.use_gpu
    model_args.batch_size = args.batch_size

    model = ParaphraseGPT(model_args)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {ckpt_path}")

    para_dev_data = load_paraphrase_data(args.para_dev)
    para_test_data = load_paraphrase_data(args.para_test, split='test')
    para_dev_dataset = ParaphraseDetectionDataset(para_dev_data, model_args)
    para_test_dataset = ParaphraseDetectionTestDataset(para_test_data, model_args)

    para_dev_loader = DataLoader(
        para_dev_dataset, shuffle=False, batch_size=args.batch_size,
        collate_fn=para_dev_dataset.collate_fn
    )
    para_test_loader = DataLoader(
        para_test_dataset, shuffle=False, batch_size=args.batch_size,
        collate_fn=para_test_dataset.collate_fn
    )

    dev_acc, dev_f1, dev_y_pred, _, dev_sent_ids = model_eval_paraphrase(
        para_dev_loader, model, device
    )
    print(f"dev paraphrase acc :: {dev_acc:.3f}, dev f1 :: {dev_f1:.3f}")

    test_y_pred, test_sent_ids = model_test_paraphrase(para_test_loader, model, device)

    label_map = {0: 3919, 1: 8505}
    os.makedirs("predictions", exist_ok=True)
    with open(args.para_dev_out, "w") as f:
        f.write("id \t Predicted_Is_Paraphrase \n")
        for sent_id, pred in zip(dev_sent_ids, dev_y_pred):
            f.write(f"{sent_id}, {label_map[pred]} \n")
    with open(args.para_test_out, "w") as f:
        f.write("id \t Predicted_Is_Paraphrase \n")
        for sent_id, pred in zip(test_sent_ids, test_y_pred):
            f.write(f"{sent_id}, {label_map[pred]} \n")
    print(f"Wrote dev predictions to {args.para_dev_out}")
    print(f"Wrote test predictions to {args.para_test_out}")


if __name__ == "__main__":
    args = get_dpo_args()
    seed_everything(11711)

    if args.test_only:
        if not args.ckpt:
            raise SystemExit("--ckpt is required when using --test_only")
        args = add_arguments(args)
        test_dpo(args, args.ckpt)
    else:
        save_path = train_dpo(args)
        test_dpo(args, save_path)