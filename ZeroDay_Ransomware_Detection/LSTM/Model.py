import torch
from torch import nn 
from transformers import AutoModel

class Classifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, model_name:str, compression_ratio:int):
        super(Classifier,self).__init__()
        
        self.llm_encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Usa hidden_size dinâmico em vez de 768 fixo.
        # hidden_size deve ser passado corretamente (896 para Qwen-0.5B, 768 para BERT)
        self.lstm = nn.LSTM(hidden_size=compression_ratio, input_size=hidden_size, num_layers=1,
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
            sub_mask = sub_mask.squeeze(1)
            
            # Passando pelo LLM (usando o novo nome llm_encoder)
            bert_out, _ = self.llm_encoder(input_ids=sub_input_id, attention_mask=sub_mask, return_dict=False)

            # Converter para float32 para compatibilidade com LSTM e evitar erro de bfloat16
            bert_out = bert_out.float()

            # O LSTM processa a saída do LLM
            bert_out_lstm, _ = self.lstm(bert_out)
            
            # Pega o último estado oculto da sequência
            concatenated_sub_tensors.append(bert_out_lstm[:, -1, :])
            
        result = torch.cat(concatenated_sub_tensors, dim=1)
        linear_output = self.fc1(result)
        return linear_output