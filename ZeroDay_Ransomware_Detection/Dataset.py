import torch
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) #Usa o tonkenizador padrão de qualquer modelo dito

tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token is None:
    # Adiciona um novo token de padding ao tokenizador
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

labels = {
        "Goodware": 0,
        "malware": 1,
            }

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        dataframe= dataframe.fillna('')
        texts = dataframe[['apiFeatures', 'dropFeatures', 'regFeatures', 'filesFeatures', 'filesEXTFeatures', 'dirFeatures', 'strFeatures']].values
        new_array = []  
        for sublist in texts:  
             # Reduzido de 512 para 128 para evitar OOM (7 features * 128 tokens = 896 tokens total)
             token_text =  tokenizer(sublist.tolist(),padding='max_length',max_length=128,truncation=True,return_tensors="pt")
             new_array.append(token_text)  
        self.texts = new_array
        self.labels = np.where((dataframe['family'] >= 1) & (dataframe['family'] <= 11), 1, dataframe['family'])
        assert len(self.texts) == len(self.labels), '[ERROR] texts count not equal label count.'

    def __getitem__(self, idx):
        text= self.texts[idx]
        label = self.labels[idx]
        return text,  label
        
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)