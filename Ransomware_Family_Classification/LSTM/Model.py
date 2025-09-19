# Arquivo: Model_LSTM.py (exemplo de nome)
import torch
from torch import nn 
from transformers import AutoModel # <-- MUDANÇA

class Classifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, model_name:str, compression_ratio:int):
        super(Classifier,self).__init__()
        # Carrega o LLM base (BERT, RoBERTa, etc.) de forma genérica
        self.llm_encoder = AutoModel.from_pretrained(model_name) # <-- MUDANÇA
        
        # A camada específica deste modelo é mantida
        # NOTA: No seu código original, aqui estava um Avgpooling. Você substituiria por sua camada LSTM.
        # Ex: self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=compression_ratio, batch_first=True)
        self.pooling = nn.AdaptiveAvgPool1d(compression_ratio) 
        
        self.fc1 = nn.Linear(compression_ratio*max_seq_len*7, num_classes)
        
    def forward(self, input_id, mask):
        # ... O método forward() seria idêntico, mas usando a camada LSTM
        # A lógica para extrair as features do LLM é a mesma.
        input_ids = torch.split(input_id, 1, dim=1)
        masks = torch.split(mask, 1, dim=1)
        concatenated_sub_tensors = []
        for sub_input_id, sub_mask in zip(input_ids, masks):
            sub_input_id = sub_input_id.squeeze(1)
            
            model_output = self.llm_encoder(input_ids=sub_input_id, attention_mask=sub_mask, return_dict=False)
            last_hidden_state = model_output[0]
            
            # Aqui você aplicaria sua lógica de compressão específica (LSTM)
            pooled_output = self.pooling(last_hidden_state)
            
            concatenated_sub_tensors.append(pooled_output)
        
        result = torch.cat(concatenated_sub_tensors, dim=1)
        batch_size = result.shape[0]
        linear_output = self.fc1(result.view(batch_size,-1))
        return linear_output