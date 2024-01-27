## データセット作成

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer
torch.manual_seed(0)

model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)

train_val_df = pd.read_csv('train_val_df.tsv', sep='\t', index_col=0)
# print(train_val_df)

class CreateDataset(Dataset):
  def __init__(self, tokenizer, df):
    self.tokenizer = tokenizer
    self.df = df

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    text = self.df.at[index, "text"]
    encoding = self.tokenizer(text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    encoding = {key: torch.squeeze(value) for key, value in encoding.items()}
    encoding["labels"] = self.df.at[index, "artist"]
    return encoding

dataset = CreateDataset(tokenizer, train_val_df)
print(len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-45, 45])
train_dataloader = DataLoader(train_dataset, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=8)
print(len(train_dataset), len(val_dataset))


## 学習

import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification
device = "cuda"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
df = pd.DataFrame(columns=["epoch","train_loss","val_loss"])

import numpy as np
EPOCH = 14
train_loss_list = []
val_loss_list = []
for epoch in range(EPOCH):
  model.train()
  train_loss = 0
  val_loss = 0
  for i, batch in enumerate(train_dataloader):
    batch = {key: value.to(device) for key, value in batch.items()}
    output = model(**batch)
    loss = output.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss += loss

  model.eval()
  labels_list, outputs_list = [], []
  for i, batch in enumerate(val_dataloader):
    batch = {key: value.to(device) for key, value in batch.items()}
    labels_list = np.concatenate([labels_list, batch["labels"].cpu().detach().numpy()])
    output = model(**batch)
    val_loss += output.loss
    output = output.logits.argmax(axis=1).cpu().detach().numpy()
    outputs_list = np.concatenate([outputs_list, output])

  train_loss_list.append(train_loss.item())
  val_loss_list.append(val_loss.item())
  accuracy = sum(outputs_list == labels_list) / len(outputs_list)
  print(f"epoch: {epoch + 1}, train_loss: {round(train_loss.item(), 1)}, val_loss: {round(val_loss.item(), 1)}, accuracy: {accuracy} {sum(outputs_list == labels_list)}/{len(outputs_list)}")
  temp_df = pd.DataFrame([[epoch + 1],[train_loss.item()],[val_loss.item()]], index=df.columns).T
  df = df.append(temp_df, ignore_index=True) 

torch.save(model, "model_wm_test.pth")
df.to_csv('loss_wm_test.tsv', sep='\t', mode="w")