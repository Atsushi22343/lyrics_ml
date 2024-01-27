## テストデータを用いたモデルの評価

## データセット作成

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer
torch.manual_seed(0)

model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)

test_df = pd.read_csv('test_df.tsv', sep='\t', index_col=0)
test_df = test_df.reset_index(drop=True)
# print(test_df)

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

test_dataset = CreateDataset(tokenizer, test_df)
print(len(test_dataset))
test_dataloader = DataLoader(test_dataset, batch_size=1)


## 学習

device = "cuda"
model = torch.load('./model_wm_test.pth')

import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

model.eval()
labels_list, outputs_list = [], []
for i, batch in enumerate(test_dataloader):
    batch = {key: value.to(device) for key, value in batch.items()}
    labels_list = np.concatenate([labels_list, batch["labels"].cpu().detach().numpy()])
    output = model(**batch)
    output = output.logits.argmax(axis=1).cpu().detach().numpy()
    outputs_list = np.concatenate([outputs_list, output])

print(classification_report(labels_list, outputs_list)) # 適合率、再現率、F値

cm = confusion_matrix(labels_list, outputs_list)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.savefig('heatmap.png') # ヒートマップで可視化
# accuracy = sum(outputs_list == labels_list) / len(outputs_list)
# print(f"accuracy: {accuracy} {sum(outputs_list == labels_list)}/{len(outputs_list)}")