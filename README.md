# GPT-2 Fine-Tuning, LoRA, and DPO for Paraphrase Detection and Sonnet Generation

Fine-tuning all the parameters of a large language model works well but is expensive. LoRA (Low-Rank Adaptation) offers a cheaper alternative by only training small low-rank matrices while keeping the rest of the model frozen. We study how LoRA compares to full fine-tuning on GPT-2 (124M parameters) on two tasks: paraphrase detection (a classification task) and sonnet generation (an open-ended generation task). We implement LoRA from scratch, adding trainable low-rank matrices to the query and value projections in each attention layer. For paraphrase detection, LoRA with just 147K trainable parameters (0.12% of the model) reaches 85.9% dev accuracy versus 88.7% for full fine-tuning, a gap of only 2.8 points. For sonnet generation, however, LoRA performs much worse: its outputs are often incoherent, scoring 17.7 chrF compared to 25.8 in our initial full fine-tuning setup, and increasing the rank from 1 to 16 does not materially improve generation quality. We further study decoding and preference-based post-training for sonnet generation. Nucleus sampling substantially outperforms beam search, and Direct Preference Optimization (DPO) provides an additional improvement on top of supervised fine-tuning, raising dev chrF from 41.5 to 42.0. Our error analysis shows that LoRA's paraphrase mistakes concentrate on high-overlap sentence pairs, while its sonnet failures stem from an inability to shift the output distribution enough toward Shakespearean language.

Report: [CS224N Final Report](https://drive.google.com/file/d/1Gfw7L5mWUlWWDuI8vb7QC9cu2pHfbfyX/view?usp=sharing)

## What We Built

- Implemented GPT-2 base components from scratch, including multi-head causal self-attention, transformer layers, and AdamW.
- Built a cloze-style paraphrase detector using GPT-2 token embeddings for the labels "yes" and "no".
- Built an autoregressive sonnet generator with full-sequence next-token training.
- Implemented LoRA from scratch by injecting low-rank adapters into the query and value projections of each attention layer.
- Implemented DPO for both tasks:
  - paraphrase DPO over preferred vs. dispreferred yes/no labels
  - sequence-level sonnet DPO over preferred vs. mismatched continuations
- Evaluated decoding strategies for sonnet generation, including nucleus sampling and beam search.

## Key Results

| Task | Method | Result |
| --- | --- | --- |
| Paraphrase detection | Full fine-tuning | 88.7% dev accuracy |
| Paraphrase detection | LoRA (r=4) | 85.9% dev accuracy |
| Paraphrase detection | DPO | 89.3% dev accuracy, 0.886 dev F1 |
| Sonnet generation | Full fine-tuning (initial setup) | 25.8 chrF |
| Sonnet generation | Full fine-tuning + tuned top-p sampling | 41.507 chrF |
| Sonnet generation | DPO from pretrained GPT-2 | 41.732 chrF |
| Sonnet generation | Full fine-tuning + DPO | **42.009 chrF** |
| Sonnet generation | LoRA (r=4) | 17.7 chrF |

Additional findings:
- LoRA used only 147K trainable parameters (0.12% of GPT-2) for paraphrase detection while staying within 2.8 points of full fine-tuning.
- For sonnet generation, beam search underperformed nucleus sampling and typically produced lower-quality, more repetitive text.
- Increasing LoRA rank from 1 to 16 did not materially improve sonnet generation quality.

## Repository Layout

- `paraphrase_detection.py`: full fine-tuning baseline for paraphrase detection
- `sonnet_generation.py`: full fine-tuning baseline for sonnet generation, plus top-p sampling and beam search
- `lora.py`: LoRA implementation
- `dpo.py`: DPO objectives for classification and sequence generation
- `dpo_paraphrase.py`: DPO training/evaluation for paraphrase detection
- `dpo_sonnet.py`: DPO training/generation for sonnet generation
- `datasets.py`: dataset loading and DPO pair construction
- `error_analysis.py`: analysis scripts used for the report
- `predictions/`: selected example outputs from final generation runs

## Example Commands

```bash
# Full fine-tuning: paraphrase detection
python paraphrase_detection.py --use_gpu

# Full fine-tuning: sonnet generation
python sonnet_generation.py --use_gpu

# DPO for paraphrase detection
python dpo_paraphrase.py --use_gpu --epochs 10

# DPO for sonnet generation (best setup)
python dpo_sonnet.py --use_gpu --ckpt_base 9_10-1e-05-sonnet.pt --epochs 5 --beta 0.15 --batch_size 2
```

## Attribution

This project builds on the Stanford CS224N default final project starter code and uses parts of the Hugging Face `transformers` stack. See `LICENSE` for the Apache 2.0 license and the report for full citations.
