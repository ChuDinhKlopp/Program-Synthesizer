import json
from typing import Union, List, Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import ProgramSynthesizer
from tokenizer import DSLSymbolTokenizer
from transformers import AutoTokenizer, AutoModel

class ProgramSynthesisDataset(Dataset):
    def __init__(self, path: str):
        with open(path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        description = self.data[idx]['text']
        function_list = self.data[idx]['functions_list']
        sample = {"description": description, "function_list": function_list}
        return sample

class ProgramSynthesisCollator:
    def __init__(self, enc_tokenizer, dec_tokenizer):
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer

    def __call__(self, batch):
        # NL Description: List of dicts
        descriptions = [self.enc_tokenizer(sample['description'], return_tensors='pt', padding='max_length', max_length=512, truncation=True) for sample in batch]
        # NL Description: Convert to dict of lists
        tokenized_descriptions = {key: [item[key] for item in descriptions] for key in descriptions[0]}
        tokenized_descriptions['input_ids'] = torch.stack(tokenized_descriptions['input_ids']).squeeze(1)
        tokenized_descriptions['attention_mask'] = torch.stack(tokenized_descriptions['attention_mask']).squeeze(1)
        # Function name: List of dicts
        function_list = [self.dec_tokenizer(sample['function_list'], return_tensors='pt', padding="max_length") for sample in batch]
        # Function name: Convert to dict of lists
        tokenized_functions = {key: [item[key] for item in function_list] for key in function_list[0]}
        tokenized_functions['input_ids'] = torch.stack(tokenized_functions['input_ids']).squeeze(1)
        tokenized_functions['attention_mask'] = torch.stack(tokenized_functions['attention_mask']).squeeze(1)

        sos_id = self.dec_tokenizer._dictionary._vocabulary['<sos>']
        eos_id = self.dec_tokenizer._dictionary._vocabulary['<eos>']
        pad_id = self.dec_tokenizer._dictionary._vocabulary['<pad>']
        # Prepare inputs and targets for decoder
        # Inputs : <sos> s1 s2 ... -> replace all <eos> with <pad>
        dec_input_ids = dict({'input_ids': tokenized_functions['input_ids'].clone(), 'attention_mask': tokenized_functions['attention_mask'].clone()})
        dec_input_ids['input_ids'][dec_input_ids['input_ids'] == eos_id] = pad_id
        # Since <eos> of input_ids has been replaced by <pad>, we have to remove the attention away from this token
        # Shift left
        dec_input_ids['attention_mask'] = torch.roll(dec_input_ids['attention_mask'], shifts=(0, -1), dims=(0, 1))
        # replace the last elements of each row with zero
        dec_input_ids['attention_mask'][:, -1] = 0
        # Targets: s1 s2 <eos> ... -> shift left -> <sos> now become the last elements -> replace all <sos> with <pad>
        target_ids = dict({'input_ids': tokenized_functions['input_ids'].clone(), 'attention_mask': tokenized_functions['attention_mask'].clone()})
        target_ids['input_ids'] = torch.roll(target_ids['input_ids'], shifts=(0, -1), dims=(0, 1))
        target_ids['input_ids'][:, -1] = pad_id
        return {"description": tokenized_descriptions, "dec_input_ids": dec_input_ids, "target_ids": target_ids}

