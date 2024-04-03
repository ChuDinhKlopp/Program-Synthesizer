import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from data_loader import ProgramSynthesisDataset, ProgramSynthesisCollator

from transformers import AutoTokenizer
from tokenizer import DSLSymbolTokenizer
from model import ProgramSynthesizer

class Trainer:
    def __init__(self, model, criterion, optimizer, device=None, debug=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = self._get_device(device)

        # send model to device
        self.model.to(self.device)
        # attributes
        self.train_losses = []
        self.val_losses = []

        self.debug = debug

    def _to_device(self, enc_inputs, dec_inputs, target_ids, device):
        enc_inputs['input_ids'] = enc_inputs['input_ids'].to(device)
        enc_inputs['attention_mask'] = enc_inputs['attention_mask'].to(device)
        dec_inputs['input_ids'] = dec_inputs['input_ids'].to(device)
        dec_inputs['attention_mask'] = dec_inputs['attention_mask'].to(device)
        target_ids['input_ids'] = target_ids['input_ids'].to(device)
        return enc_inputs, dec_inputs, target_ids

    def _get_device(self, device):
        if device is None:
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            dev = device
        return dev

    def _train(self, loader):
        # put model in train mode
        self.model.train()

        for i, batch in enumerate(loader):
            enc_inputs, dec_inputs, target_ids = self._to_device(batch['description'], batch['dec_input_ids'], batch['target_ids'], self.device)
            if self.debug:
                print(f"trainer.py dec_inputs: {dec_inputs}")
                print(f"trainer.py enc_inputs: {enc_inputs}")
            # forward pass
            preds = self.model(enc_inputs=enc_inputs, dec_inputs=dec_inputs)

            print(f"trainer.py raw preds: {preds}")
            print(f"trainer.py argmax preds: {torch.argmax(preds, dim=2)}")
            print(f"trainer.py targets: {target_ids['input_ids']}")
            # loss
            loss = self._compute_loss(preds, target_ids['input_ids'])
            print(f"Batch {i} - Training loss: {loss}")
            # backprop
            loss.backward()
            # update parameters
            self.optimizer.step()
            #break

        return loss.item()


    def _validate(self, loader):
        # put model in validation mode
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                enc_inputs, dec_inputs, target_ids = self._to_device(batch['description'], batch['dec_input_ids'], batch['target_ids'], self.device)
                # forward pass
                preds = self.model(enc_inputs=enc_inputs, dec_inputs=dec_inputs)
                loss = self._compute_loss(preds, target_ids['input_ids'])
                print(f"Batch {i} - Validation loss: {loss}")
        return loss.item()

    def _compute_loss(self, preds, targets):
        loss = self.criterion(preds.transpose(1, 2), targets)
        return loss

    def fit(self, train_loader, val_loader, epochs):
        start_training_time = time.time()
        for i, epoch in enumerate(range(epochs)):
            print(f"Epoch {i}")
            # train
            train_loss = self._train(train_loader)
            # validate
            val_loss = self._validate(val_loader)
            #break
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

        total_training_time = time.time() - start_training_time
        self.model.save_checkpoint()


if __name__ == "__main__":
    enc_tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    dec_tokenizer = DSLSymbolTokenizer()

    collator = ProgramSynthesisCollator(enc_tokenizer=enc_tokenizer, dec_tokenizer=dec_tokenizer)
    train_dataset = ProgramSynthesisDataset("./data/train_2.json")
    val_dataset = ProgramSynthesisDataset("./data/val_2.json")
    print(f"trainer.py train: {len(train_dataset)} - val: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=collator)

    model = ProgramSynthesizer(d_model=6, n_head=1, max_sequence_len=7, n_layers=2, name='model_v1.0', ckpt_dir="./ckpt", debug=True)

    continue_from_ckpt = input("Continue training from checkpoint [y/n]? ")
    if continue_from_ckpt == 'y':
      ckpt_name = input("Enter checkpoint name: ")
      model.load_checkpoint(ckpt_name)
      lr = float(input("Continue training with learning rate: "))
    elif continue_from_ckpt == 'n':
      lr = 5e-5

    criterion = nn.CrossEntropyLoss(ignore_index=dec_tokenizer._dictionary._vocabulary['<pad>'])
    optimizer = AdamW(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=device, debug=False)
    trainer.fit(train_loader, val_loader, epochs=10)
