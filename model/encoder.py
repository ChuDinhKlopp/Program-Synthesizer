import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Union, Optional, List

class Encoder(nn.Module):
    def __init__(self, checkpoint: str, max_sequence_len: int=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)
        self.max_sequence_len = max_sequence_len

    def forward(self, text: Union[str, List[str]], padding=False):
        if padding:
            inputs = self.tokenizer(text, return_tensors='pt', padding="max_length", max_length=self.max_sequence_len)
            outputs = self.model(**inputs).last_hidden_state
        else:
            inputs = self.tokenizer(text, return_tensors='pt')
            outputs = self.model(**inputs).last_hidden_state
        return outputs
