import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProduct(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1, debug=False):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.debug = debug

    def forward(self, q, k ,v, mask=None):
        """
        Args:
            q, k, v are of the size (batch_size * num_head * sequence_len * embed_dim_of_one_head)
        Returns:
            output: contextualized tokens
            attn: dot-product of queries and keys 
        """
        # Q.K/d_k
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) # (bacth_size * num_head * sequence_len * sequence_len)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            if self.debug:
                print(f"attention.py mask in scaled_dot_product: {mask}")
                print(f"attention.py attn.masked_fill: {attn}")

        attn = self.dropout(F.softmax(attn, dim=-1))
        #attn = F.softmax(attn, dim=-1)
        if self.debug:
            print(f"attention.py attention after softmax and dropout: {attn}")
        if mask is not None:
            attn = attn.masked_fill(mask == 0, 0)
            if self.debug:
                print(f"attention.py masked attn after softmax: {attn}")
        # softmax(Q.K/d_k).V
        output = torch.matmul(attn, v)
        if self.debug:
            print(f"attention.py softmax(qk)v: {output}")

        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, n_layers, d_k, d_v, dropout=0.1, debug=False):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, d_k * n_head, bias=False)
        self.w_k = nn.Linear(d_model, d_k * n_head, bias=False)
        self.w_v = nn.Linear(d_model, d_v * n_head, bias=False)
        self.fc = nn.Linear(d_v * n_head, d_model, bias=True)

        # Weight initialization
        nn.init.kaiming_uniform_(self.w_q.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.w_k.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.w_v.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.02/math.sqrt(2 * n_layers))

        self.scaled_dot_product = ScaledDotProduct(temperature=d_k**0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.debug = debug

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q, k, v are of the size (batch_size * sequence_len * d_model)
        Returns:
            q: contextualized tokens + residual
        """
        batch_size = q.size(0)
        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v
        len_q, len_k, len_v = q.size(1), k.size(1), v.size(1)
        
        residual_q = q
        if self.debug:
            print(f"attention.py q before all the attention: {q}")

            print(f"attention.py w_q: {self.w_q.weight.grad}")
            print(f"attention.py w_k: {self.w_k.weight.grad}")
            print(f"attention.py w_v: {self.w_v.weight.grad}")

        q = self.w_q(q).view(batch_size, len_q, n_head, d_q)
        k = self.w_k(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_v(v).view(batch_size, len_v, n_head, d_v)

        q = q.transpose(1, 2) # (batch_size, n_head, len_q, d_q)
        k = k.transpose(1, 2) # (batch_size, n_head, len_k, d_k)
        v = v.transpose(1, 2) # (batch_size, n_head, len_v, d_v)
        if self.debug:
            print(f"attention.py query = {q}")
            print(f"attention.py key = {k}")
            print(f"attention.py value = {v}")
        
        q, attn = self.scaled_dot_product(q, k, v, mask=mask) # (batch_size, n_head, len_v, d_v)

        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1) # (batch_size, len_q, (n_head * d_q)) 
        # Prevent overfitting
        q = self.dropout(self.fc(q)) # (batch_size, n_head, d_model)
        # Add and norm
        # Add residual connection
        # Post-norm
        #q += residual_q
        #if self.debug:
        #    print(f"attention.py softmax(qk)v + residual: {q}")
        ## Layer normalization
        #q = self.layer_norm(q)
        # Pre-norm
        q = self.layer_norm(q)
        if self.debug:
            print(f"attention.py softmax(qk)v + norm: {q}")
        # Layer normalization
        q += residual_q
        if self.debug:
            print(f"attention.py normalized softmax(qk)v + residual: {q}")

        return q, attn

