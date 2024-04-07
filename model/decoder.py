import copy
import math
import torch
import torch.nn as nn
from attention import MultiHeadAttention
from symbol_embeddings import Embedding, PositionalEmbedding

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, n_layers, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_hid)
        self.fc3 = nn.Linear(d_hid, d_in)
        # Weight initialization
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.02/math.sqrt(2 * n_layers))

        self.layernorm = nn.LayerNorm(d_in, eps=1e-12)
        # using dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: size(batch_size, max_len, d_model)
        Returns:
            x: size(batch_size, max_len, d_model)
        """
        residual_x = x # (batch_size, max_len, d_model)
        x = self.gelu(self.fc1(x)) # (batch_size, max_len, d_hid)
        x = self.gelu(self.fc2(x))
        x = self.fc3(x) # (batch_size, max_len, d_model)
        # Prevent overfitting
        x = self.dropout(x)
        # Add and norm
        # Post-norm
       # # Add residual connection
       # x += residual_x
       # # Layer normalization
       # x = self.layernorm(x)
        # Pre-norm
        # Layer normalization
        x = self.layernorm(x)
        # Add residual connection
        x += residual_x
        return x

class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, n_layers, d_k, d_v, dropout=0.1, debug=False):
        super().__init__()
        self.masked_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, n_layers=n_layers, d_k=d_k, d_v=d_v, dropout=dropout, debug=debug)
        self.cross_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, n_layers=n_layers, d_k=d_k, d_v=d_v, dropout=dropout, debug=debug)
        self.ffn = PositionwiseFeedForward(d_in=d_model, d_hid=d_model*8, n_layers=n_layers, dropout=dropout)

    def forward(self, decoder_input, encoder_output, slf_attn_mask=None, cross_attn_mask=None):
        dec_output, dec_slf_attn = self.masked_attn(q=decoder_input, k=decoder_input, v=decoder_input, mask=slf_attn_mask)
        dec_output, enc_dec_cross_attn = self.cross_attn(q=dec_output, k=encoder_output, v=encoder_output, mask=cross_attn_mask)
        dec_output = self.ffn(dec_output)

        return dec_output, dec_slf_attn, enc_dec_cross_attn

class Decoder(nn.Module):
    def __init__(self, d_model, max_sequence_len, n_layers, vocab_size, pad_idx, n_head, d_k, d_v, dropout=0.1, debug=False):
        super().__init__()
        self.word_emb = Embedding(vocab_size=vocab_size, embed_dim=d_model, padding_idx=pad_idx)
        self.pos_emb = PositionalEmbedding(max_seq_len=max_sequence_len, embed_model_dim=d_model, debug=debug)
        decoder_layer = DecoderLayer(n_head=n_head, d_model=d_model, n_layers=n_layers, d_k=d_k, d_v=d_v, debug=debug, dropout=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(n_head=n_head, d_model=d_model, n_layers=n_layers, d_k=d_k, d_v=d_v, debug=debug)
            # copy.deepcopy(decoder_layer)
            for _ in range(n_layers)])
        # Handle Add and Norm between decoder layers
        self.norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.n_head = n_head
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.debug = debug

    def forward(self, enc_out, input_ids, attention_mask):
        """
        Args:
            dec_in: tokenized sequences of size (batch_size, max_sequence_len)
        """
        attention_mask = attention_mask.unsqueeze(1)
        # attention_mask = torch.matmul(attention_mask.transpose(1, 2), attention_mask)
        attention_mask = attention_mask.repeat(1, attention_mask.size(2), 1)
        attention_mask = attention_mask.unsqueeze(1).expand(-1, self.n_head, -1, -1) # add n_head dimension to the mask so it matches the decoder input dimension
        #causal_mask = torch.full((input_ids.size(0), self.n_head, input_ids.size(1), input_ids.size(1)), float(1))
        causal_mask = torch.tril(attention_mask, diagonal=0).to(self.device)
        # Word embedding + Positional Embedding
        x = self.pos_emb(self.word_emb(input_ids)) # (batch_size, max_sequence_len, d_model)
        if self.debug:
            print(f"decoder.py x after pos+word-emb: {x}")
        # Dropout + LayerNorm
        x = self.norm(x)
        if self.debug:
            print(f"decoder.py x after dropout and layernorm: {x}")
        x = self.dropout(x)
        if self.debug:
            print(f"decoder.py x after dropout: {x}")
        # Decoder layer stack
        for i, layer in enumerate(self.layer_stack):
            if self.debug:
                print(f"decoder.py step {i}")
            x, _, _ = layer(decoder_input=x, encoder_output=enc_out, slf_attn_mask=causal_mask, cross_attn_mask=None)
        
        return {"last_hidden_state": x}
