from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    TODO: Implement this helper method for the forward function.
      - This function is applied after the multi-head attention layer as well as after the feed forward layer.
      - GPT-2 layer applies dropout to the transformed output of each sub-layer,
        before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
        IN THIS FUNCTION.
    """
    # this implements the residual connection pattern from figure 2
    # 1. linear projection on the sublayer output
    # 2. dropout for regularization
    # 3. add back the original input (the "skip connection")
    # note: we do NOT apply layernorm here. that happens in forward() before each sublayer
    output = dense_layer(output)
    output = dropout(output)
    output = output + input
    return output


  def forward(self, hidden_states, attention_mask):
    # GPT-2 uses "pre-norm": layernorm goes BEFORE each sublayer, not after
    # this is different from the original transformer paper which uses post-norm
    # see figure 2 in the handout for the diagram

    # multi-head attention sublayer
    # step 1: normalize, then run attention, then residual connection
    normalized_for_attn = self.attention_layer_norm(hidden_states)
    attn_output = self.self_attention(normalized_for_attn, attention_mask)
    # add() does: project -> dropout -> add residual (see above)
    hidden_states = self.add(hidden_states, attn_output, self.attention_dense, self.attention_dropout)

    # feed-forward sublayer
    # step 2: normalize, run through ffn (expand dim with gelu), then residual
    # the ffn expands 768 -> 2304 (interm_dense), applies gelu, then add() projects back 2304 -> 768
    normalized_for_ffn = self.out_layer_norm(hidden_states)
    ffn_output = self.interm_af(self.interm_dense(normalized_for_ffn))
    hidden_states = self.add(hidden_states, ffn_output, self.out_dense, self.out_dropout)

    return hidden_states

