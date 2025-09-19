import torch
from torch import nn 
from transformers import AutoModel # <-- MUDANÇA

class Classifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, model_name:str, compression_ratio:int):
        super(Classifier,self).__init__()
        self.llm_encoder = AutoModel.from_pretrained(model_name) # Carrega o LLM base de forma genérica

        self.pooling = nn.AdaptiveMaxPool1d(compression_ratio)
        
        self.fc1 = nn.Linear(compression_ratio*max_seq_len*7, num_classes)
        
    def forward(self, input_id, mask):
        input_ids = torch.split(input_id, 1, dim=1)
        masks = torch.split(mask, 1, dim=1)
        concatenated_sub_tensors = []
        for sub_input_id, sub_mask in zip(input_ids, masks):
            sub_input_id = sub_input_id.squeeze(1)
            
            model_output = self.llm_encoder(input_ids=sub_input_id, attention_mask=sub_mask, return_dict=False) #Usa o encoder genérico
            last_hidden_state = model_output[0] # Pega a saída principal
            
            # Aplica o pooling específico deste modelo
            pooled_output = self.pooling(last_hidden_state)
            
            concatenated_sub_tensors.append(pooled_output)
        
        result = torch.cat(concatenated_sub_tensors, dim=1)
        batch_size = result.shape[0]
        linear_output = self.fc1(result.view(batch_size,-1))
        return linear_output