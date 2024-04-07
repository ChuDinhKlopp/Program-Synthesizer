import os

import torch
import torch.nn as nn

from transformers import AutoModel
from decoder import Decoder
from tokenizer import DSLDictionary

class ProgramSynthesizer(nn.Module):
    def __init__(self, name, d_model=768, n_head=1, max_sequence_len=512, n_layers=5, debug=False, ckpt_dir='/ckpt'):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_len = max_sequence_len

        self._dictionary = DSLDictionary()
        # Encoder
        self.encoder = AutoModel.from_pretrained("distilbert/distilbert-base-uncased")
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.enc_dim_reduction = nn.Sequential(
            nn.Linear(self.encoder.config.max_position_embeddings * self.encoder.config.dim, 256),
            nn.ReLU(),
            nn.Linear(256, max_sequence_len * d_model)
        )
        # self.enc_dim_reduction.apply(self.weights_init)
        # Decoder
        self.decoder = Decoder(d_model=d_model, max_sequence_len=max_sequence_len, n_layers=n_layers, vocab_size=len(self._dictionary._vocabulary), pad_idx=self._dictionary._vocabulary['<pad>'], n_head=n_head, d_k=int(d_model/n_head), d_v=int(d_model/n_head), debug=debug, dropout=0.01)
        # Classification head
        self.linear_head = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model*4), # out_features = vocab size
            nn.ReLU(),
            nn.Linear(in_features=d_model*4, out_features=d_model*4),
            nn.ReLU(),
            nn.Linear(in_features=d_model*4, out_features=len(self._dictionary._vocabulary))
        )
        # Utils
        self.name = name
        self.ckpt_dir = ckpt_dir
        self.ckpt_file = os.path.join(self.ckpt_dir, self.name + '.pt')
        self.debug = debug
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, enc_inputs, dec_inputs):
        enc_out = self.encoder(**enc_inputs).last_hidden_state
        enc_out = enc_out.reshape((-1, self.encoder.config.max_position_embeddings * self.encoder.config.dim))
        enc_out = self.enc_dim_reduction(enc_out)
        enc_out = enc_out.reshape((-1, self.max_sequence_len, self.d_model))
        if self.debug:
            print(f"model.py enc_out: {enc_out}")
        last_hidden_state = self.decoder(enc_out, **dec_inputs)['last_hidden_state'] # (batch_size, max_sequence_len, d_model)
        logits = self.linear_head(last_hidden_state) # (batch_size, max_sequence_len, vocab_size)
        return logits

    def save_checkpoint(self):
        print("---saving checkpoint---")
        torch.save(self.state_dict(), self.ckpt_file)

    def load_checkpoint(self, name):
        print("---loading checkpoint---")
        self.load_state_dict(torch.load(os.path.join(self.ckpt_dir, name + '.pt')))
