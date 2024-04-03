import torch
from model import ProgramSynthesizer
from tokenizer import DSLSymbolTokenizer
from transformers import AutoTokenizer

def _to_device(enc_inputs, dec_inputs, device):
    enc_inputs['input_ids'] = enc_inputs['input_ids'].to(device)
    enc_inputs['attention_mask'] = enc_inputs['attention_mask'].to(device)
    dec_inputs['input_ids'] = dec_inputs['input_ids'].to(device)
    dec_inputs['attention_mask'] = dec_inputs['attention_mask'].to(device)
    return enc_inputs, dec_inputs

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    dec_tokenizer = DSLSymbolTokenizer()
    model = ProgramSynthesizer(name='model_v1.0', d_model=128, n_head=1, max_sequence_len=100, n_layers=5, ckpt_dir='/content/ckpt').to(device)
    model.load_checkpoint("model_v1.0")
    inputs = ['<sos>']
    context = "The code snippet defines a neural network with a single layer, a max pooling layer with a kernel size of 3. The layer spacing is set to 0.25.\n\nIn natural language, this can be described as:\n\n\"The neural network consists of one layer, which is a max pooling layer. The max pooling layer has a kernel size of 3, which means it will divide the input data into non-overlapping blocks of size 3x3 and then apply the maximum pooling operation to each block. The output of the layer will have the same shape as the input, but with the spatial dimensions reduced by a factor of 3. Additionally, the layer spacing is set to 0.25, which means that the output of the layer will be shifted by 0.25 units in both the x and y directions, resulting in a stride of 0.5.\""
    dec_inputs = dec_tokenizer(inputs, return_tensors='pt', padding='max_length')
    enc_inputs = enc_tokenizer(context, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
    enc_inputs, dec_inputs = _to_device(enc_inputs, dec_inputs, device)
    print(torch.argmax(model(enc_inputs, dec_inputs).softmax(dim=1)))

