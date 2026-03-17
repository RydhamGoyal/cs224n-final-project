import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    # q, k, v are each [bs, num_heads, seq_len, head_size]
    # attention_mask is [bs, 1, 1, seq_len]

    # following equation (1) from the handout: Attention(Q,K,V) = softmax(QK^T / sqrt(dk)) V

    # first we need dk for the scaling factor
    dk = query.size(-1)

    # QK^T / sqrt(dk) — this gives us a score for how much each token
    # should pay attention to every other token
    # shape goes from [bs, heads, seq, head_size] to [bs, heads, seq, seq]
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (dk ** 0.5)

    # now we need the causal mask. this is what makes GPT a decoder.
    # each token should only be able to look at tokens before it (and itself)
    # triu with diagonal=1 gives us the upper triangle = future positions
    # we set those to -inf so softmax turns them into 0
    seq_len = query.size(2)
    future_mask = torch.triu(
      torch.ones(seq_len, seq_len, device=attn_scores.device), diagonal=1
    ).bool()
    attn_scores = attn_scores.masked_fill(future_mask, float('-inf'))

    # also mask out padding tokens. attention_mask has -10000 for pads, 0 for real tokens
    attn_scores = attn_scores + attention_mask

    # softmax converts scores to probabilities (they sum to 1 across each row)
    # then dropout for regularization
    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_probs = self.dropout(attn_probs)

    # multiply the attention probabilities by the values
    # this is the "weighted sum" part — each token collects info from other tokens
    context = torch.matmul(attn_probs, value)

    # finally, concat all the heads back together
    # [bs, num_heads, seq_len, head_size] -> [bs, seq_len, num_heads * head_size]
    # num_heads * head_size = hidden_size (e.g. 12 * 64 = 768)
    context = rearrange(context, 'b h t d -> b t (h d)')

    return context


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
