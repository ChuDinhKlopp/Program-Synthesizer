import logging
import torch
from typing import Union, Optional, List, Dict

class DSLDictionary:
    def __init__(self):
        self._vocabulary = {
                "Convolutional2DLayer": 0,
                "FeedForwardLayer": 1, 
                "MaxPooling2DLayer": 2,
                "<sos>": 3,
                "<eos>": 4,
                "<pad>": 5
                }
        self.key_list = list(self._vocabulary.keys())
        self.value_list = list(self._vocabulary.values())
        if self.isValid() is False:
            return None

    def isValid(self):
        if len(self.value_list) != len(set(self.value_list)):
            logging.error("Cannot initialize dictionary due to duplicate IDs assigned to different words! Please make sure the word-ID pairs are unique.")
            return False
        
        print("Dictionary initialized successfully!")
        return True

class DSLSymbolTokenizer:
    def __init__(self):
        self._dictionary = DSLDictionary()
        self._max_sequence_len = 7

    def __call__(self, inputs: Union[List[str], List[List[str]]], return_tensors: Optional[str]=None, padding: Optional[str]=None) -> Union[Dict[str, List[int]], Dict[str,  torch.Tensor]]:
        """
        Args:
            inputs: sequence of DSL symbols presented in string format, or
               list of sequences of DSL symbols
            return_tensors: if 'pt', return a torch.Tensor, if None, return a list
            padding: add <pad> tokens to list of lists to the length of the longest row
        Returns:
            tokens: sequence of tokenized DSL symbols and special tokens
        """
        if isinstance(inputs, list):
            if all(isinstance(item, str) for item in inputs):
                tokens = []
                # Tokenize and append <sos> with <eos> to the sequence
                tokens.append(self._dictionary._vocabulary['<sos>'])
                for string in inputs:
                    token = self._dictionary._vocabulary[string]
                    tokens.append(token)
                tokens.append(self._dictionary._vocabulary['<eos>'])
                attention_mask = [1] * len(tokens)
                if padding is not None:
                    if padding == 'longest':
                        tokens = tokens
                        attention_mask = attention_mask
                    if padding == 'max_length':
                        max_len = self._max_sequence_len
                        tokens = tokens + [self._dictionary._vocabulary['<pad>']]*(max_len - len(tokens))
                        attention_mask = attention_mask + [0] * (max_len - len(attention_mask))
                # Convert tokenized sequence to tensor
                if return_tensors is not None:
                    # Pytorch tensor
                    if return_tensors == 'pt':
                        tokens = torch.LongTensor(tokens)
                        attention_mask = torch.Tensor(attention_mask).unsqueeze(0)
                    else:
                        logging.error("DSLSymbolTokenizer only supports Pytorch tensor, please try again with return_tensors='pt' instead!")
                        return
                return {'input_ids': tokens, 'attention_mask': attention_mask}
            elif all(isinstance(item, list) for item in inputs):
                if all(isinstance(subitem, str) for item in inputs for subitem in item):
                    batch_input_ids = []
                    batch_attention_mask = []
                    for sequence in inputs:
                        tokens = []
                        # Tokenize and append <sos> with <eos> to the sequence
                        tokens.append(self._dictionary._vocabulary['<sos>'])
                        for string in sequence:
                            token = self._dictionary._vocabulary[string]
                            tokens.append(token)
                        tokens.append(self._dictionary._vocabulary['<eos>'])
                        attention_mask = [1] * len(tokens)
                        batch_input_ids.append(tokens)
                        batch_attention_mask.append(attention_mask)
                    # Add <pad> tokens
                    if padding is not None:
                        # Pad to the length of the longest sequence in the batch
                        if padding == 'longest':
                            longest_len = max(len(row) for row in batch)
                            batch_input_ids = [row + [self._dictionary._vocabulary['<pad>']]*(longest_len - len(row)) for row in batch_input_ids]
                            batch_attention_mask = [row + [0] * (longest_len - len(row)) for row in batch_attention_mask]
                        # Pad to the length of max sequence length
                        elif padding == 'max_length':
                            max_len = self._max_sequence_len
                            batch_input_ids = [row + [self._dictionary._vocabulary['<pad>']]*(max_len - len(row)) for row in batch_input_ids]
                            batch_attention_mask = [row + [0] * (max_len - len(row)) for row in batch_attention_mask]
                        else:
                            logging.error("Invalid padding type. Please try again with padding='longest' or padding='max_len'")

                    # Convert tokenized sequences to tensor
                    if return_tensors is not None:
                        # Pytorch tensor
                        if return_tensors == 'pt':
                            batch_input_ids = torch.LongTensor(batch_input_ids)
                            batch_attention_mask = torch.Tensor(batch_attention_mask)
                        else:
                            logging.error("DSLSymbolTokenizer only supports Pytorch tensor, please try again with return_tensors='pt' instead!")
                    return {'input_ids': batch_input_ids, 'attention_mask': batch_attention_mask}
        else:
            logging.error("Input type not supported. Please make sure the input is in correct format: List[str] or List[List[str]]")

    def decode(self, ids: Union[List[int], List[List[int]]]):
        if isinstance(ids, list):
            if all(isinstance(item, int) for item in ids):
                decoded_sequence = []
                for id in ids:
                    word_pos = self._dictionary.value_list.index(id)
                    word = self._dictionary.key_list[word_pos]
                    decoded_sequence.append(word)
                return {"decoded": decoded_sequence}
            elif all(isinstance(item, list) for item in ids):
                if all(isinstance(subitem, int) for item in ids for subitem in item):
                    decoded_batch = []
                    for sequence in ids:
                        decoded_sequence = []
                        for id in sequence:
                            word_pos = self._dictionary.value_list.index(id)
                            word = self._dictionary.key_list[word_pos]
                            decoded_sequence.append(word)
                        decoded_batch.append(decoded_sequence)
                    return {"decoded": decoded_batch}

        else:
            logging.error("Input type not supported. Please make sure the input is in correct format: List[int] or List[List[int]]")

