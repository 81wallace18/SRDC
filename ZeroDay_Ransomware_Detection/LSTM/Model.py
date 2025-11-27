import torch
from torch import nn 
from transformers import BertModel, BertTokenizer # Alterado para BERT

class Classifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, bert_model_name:str, compression_ratio:int):
        super(Classifier,self).__init__()
        # Carrega o modelo BERT
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        
        # input_size=768 assume BERT Base. 
        self.lstm = nn.LSTM(hidden_size=compression_ratio, input_size=768, num_layers=1,
                            batch_first=True)
        self.fc1 = nn.Linear(compression_ratio*7, num_classes)
        
    def forward(self, input_id, mask):
        """
        Args:
                input_id: encoded inputs ids of sent.
        """
        input_ids = torch.split(input_id, 1, dim=1)
        masks = torch.split(mask, 1, dim=1)
        concatenated_sub_tensors = []
        for sub_input_id, sub_mask in zip(input_ids, masks):
            
            sub_input_id = sub_input_id.squeeze(1)
            
            # Passando pelo BERT
            bert_out, _ = self.bert_model(input_ids=sub_input_id, attention_mask=sub_mask, return_dict=False)

            # O LSTM processa a saída do BERT
            bert_out_lstm, _ = self.lstm(bert_out)
            
            # Pega o último estado oculto da sequência
            concatenated_sub_tensors.append(bert_out_lstm[:, -1, :])
            
        result = torch.cat(concatenated_sub_tensors, dim=1)
        linear_output = self.fc1(result)
        return linear_output