import torch
from transformers import BertTokenizer
import pandas as pd
import numpy as np

# 1. Carrega o Tokenizador do BERT
# 'bert-base-uncased' é o padrão para textos em inglês (converte tudo para minúsculo)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Nota: O BERT já possui um pad_token padrão ([PAD]), então não precisamos definir manualmente.

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
            token_text = tokenizer(sublist.tolist(),padding='max_length',max_length=512,truncation=True,return_tensors="pt") 
            #o Tensor máximo do bert é 512
            new_array.append(token_text)
        self.texts = new_array
        self.labels = dataframe['family'].values
        assert len(self.texts) == len(self.labels), '[ERROR] texts count not equal label count.'

    def __getitem__(self, idx):
        text= self.texts[idx]
        label = self.labels[idx]
        return text,  label
        
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)


# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, dataframe):
#         dataframe = dataframe.fillna('')
#         texts = dataframe[['apiFeatures', 'dropFeatures', 'regFeatures', 'filesFeatures', 'filesEXTFeatures', 'dirFeatures', 'strFeatures']].values
#         new_array = []  
#         for sublist in texts:
#              # 2. Ajuste de max_length
#              # O BERT padrão suporta no máximo 512 tokens de posição. 
#              # Se mantiver 1024 aqui, o modelo (Model.py) vai dar erro na hora do forward.
#              token_text = tokenizer(
#                  sublist.tolist(),
#                  padding='max_length',
#                  max_length=512,  # AJUSTADO PARA O LIMITE DO BERT
#                  truncation=True,
#                  return_tensors="pt"
#              )
#              new_array.append(token_text)  
#         self.texts = new_array
        
#         # Mantive sua lógica de transformar famílias 1-11 em classe 1 (Malware)
#         self.labels = np.where((dataframe['family'] >= 1) & (dataframe['family'] <= 11), 1, dataframe['family'])
#         assert len(self.texts) == len(self.labels), '[ERROR] texts count not equal label count.'

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         label = self.labels[idx]
#         return text, label
        
#     def classes(self):
#         return self.labels
    
#     def __len__(self):
#         return len(self.labels)