import math
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx=None):
        """
        Args:
            vocab_size: size of DSL vocabulary
            embed_dim: dimension of embeddings
        """
        super().__init__()
        if padding_idx is not None:
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx)
        else:
            self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        """
        Args:
            x: input vectors
        Returns:
            out: embediding vector (batch_size, embed_dim)
        """
        out = self.embed(x)
        return out

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim, debug=False):
        """
        Args:
            max_seq_len: length of input sequence
            embed_model_dim: dimension of embedding (d_model)
        """
        super().__init__()
        self.embed_dim = embed_model_dim
        pos_emb = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pos_emb[pos, i] = math.sin(pos / (10000**((2*i)/self.embed_dim)))

                pos_emb[pos, i+1] = math.cos(pos / (10000**((2*(i+1))/self.embed_dim)))
        pos_emb = pos_emb.unsqueeze(0)
        self.register_buffer('pos_emb', pos_emb)
        
        self.debug = debug

    def forward(self, x):
        """
        Args:
            x: input vector - word embedding vector
        Returns:
            x: output - word embeddings + positional embeddings
        """
        if self.debug:
            print(f"symbol_embeddings.py x: {x}")
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        if self.debug:
            print(f"symbol_embeddings.py larger x: {x}")

        # add constant to embedding
        seq_len = x.size(1)
        if self.debug:
            print(f"symbol_embeddings.py x.size: {x.shape}")
        x = x + torch.autograd.Variable(self.pos_emb[:, :seq_len], requires_grad=False)
        print(f"symbol_embeddings.py x.requires_grad: {x.requires_grad}")
        if self.debug:
            print(f"symbol_embeddings.py pos+word x: {x}")
        return x
