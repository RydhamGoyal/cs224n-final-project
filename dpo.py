"""
Direct Preference Optimization (DPO) for paraphrase detection.
Reference: https://arxiv.org/abs/2305.18290
"""

import torch
import torch.nn.functional as F

def dpo_loss_paraphrase(logits_theta, logits_ref, y_w, y_l, beta=0.1):
    """
    DPO loss for cloze-style paraphrase (yes/no single-token outputs).

    logits_theta: [batch, 2] - policy model logits (index 0=no, 1=yes)
    logits_ref: [batch, 2] - reference model logits (frozen)
    y_w: [batch] - preferred label (0 or 1)
    y_l: [batch] - dispreferred label (0 or 1)
    beta: temperature (higher = stronger preference signal)
    """
    # Log probs: log pi(y|x)
    log_theta_w = F.log_softmax(logits_theta, dim=-1)[torch.arange(logits_theta.size(0)), y_w]
    log_theta_l = F.log_softmax(logits_theta, dim=-1)[torch.arange(logits_theta.size(0)), y_l]
    log_ref_w = F.log_softmax(logits_ref, dim=-1)[torch.arange(logits_ref.size(0)), y_w]
    log_ref_l = F.log_softmax(logits_ref, dim=-1)[torch.arange(logits_ref.size(0)), y_l]

    # DPO logit: beta * (log pi_theta(y_w) - log pi_ref(y_w) - log pi_theta(y_l) + log pi_ref(y_l))
    logits_dpo = beta * (log_theta_w - log_ref_w - log_theta_l + log_ref_l)
    loss = -F.logsigmoid(logits_dpo).mean()
    return loss


def dpo_loss_sonnet(log_theta_w, log_theta_l, log_ref_w, log_ref_l, beta=0.1):
    """
    DPO loss for sequence generation (sonnets).
    log_theta_w, log_theta_l, log_ref_w, log_ref_l: [batch] - log p(completion|prompt) scalars.
    """
    logits_dpo = beta * (log_theta_w - log_ref_w - log_theta_l + log_ref_l)
    loss = -F.logsigmoid(logits_dpo).mean()
    return loss
