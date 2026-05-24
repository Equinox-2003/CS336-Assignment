import torch
from torch import nn
from transformer_block import transformer_block
from Embedding import Embedding
from Linear import Linear
from RoPE import RoPE
from RMSNorm import RMSNorm
from SwiGLUFFN import SwiGLUFFN
from softmax import softmax

class Transformer_LM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, torch.Tensor],
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.weights = weights

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        token_embedding = Embedding(self.vocab_size, self.d_model)
        token_embedding.load_state_dict({'embeddings': self.weights['token_embeddings.weight']})

        x = token_embedding(in_indices)

        for i in range(self.num_layers):
            attn_q_proj_weight = self.weights[f"layers.{i}.attn.q_proj.weight"]
            attn_k_proj_weight = self.weights[f"layers.{i}.attn.k_proj.weight"]
            attn_v_proj_weight = self.weights[f"layers.{i}.attn.v_proj.weight"]
            attn_o_proj_weight = self.weights[f"layers.{i}.attn.output_proj.weight"]
            ln1_weight = self.weights[f"layers.{i}.ln1.weight"]
            ln2_weight = self.weights[f"layers.{i}.ln2.weight"]
            ffn_w1_weight = self.weights[f"layers.{i}.ffn.w1.weight"]
            ffn_w2_weight = self.weights[f"layers.{i}.ffn.w2.weight"]
            ffn_w3_weight = self.weights[f"layers.{i}.ffn.w3.weight"] 

            trans_block = transformer_block(self.d_model, self.num_heads, self.d_ff, self.context_length, self.rope_theta, attn_q_proj_weight, attn_k_proj_weight, attn_v_proj_weight, attn_o_proj_weight, ln1_weight, ln2_weight, ffn_w1_weight, ffn_w2_weight, ffn_w3_weight) 
            x = trans_block(x)
        
        rms_norm = RMSNorm(self.d_model)
        rms_norm.load_state_dict({'g': self.weights["ln_final.weight"]})
        x = rms_norm(x)

        linear_layer = Linear(self.d_model, self.vocab_size)
        linear_layer.load_state_dict({'W': self.weights['lm_head.weight']})
        x = linear_layer(x)

        # handout说最终输出logits即可
        return x
    