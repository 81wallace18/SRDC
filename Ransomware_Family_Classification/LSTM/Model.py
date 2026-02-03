import torch
from torch import nn 
from transformers import AutoModel 

class Classifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, model_name:str, compression_ratio:int):
        super(Classifier,self).__init__()
        # Carrega o LLM base (BERT, RoBERTa, etc.) de forma genérica
        self.llm_encoder = AutoModel.from_pretrained(model_name)
        
        # Camada de pooling para transformar a saída do BERT em um único vetor por amostra.
        # Ele irá calcular a média dos vetores de todos os tokens na sequência.
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Camada linear para a classificação final. A entrada é o hidden_size do BERT.
        self.fc1 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_id, mask):
        # Passa a sequência inteira (input_id) e a máscara para o encoder.
        # Não há necessidade de um loop.
        # Shape da saída (last_hidden_state): [batch_size, sequence_length, hidden_size]
        model_output = self.llm_encoder(input_ids=input_id, attention_mask=mask, return_dict=False)
        last_hidden_state = model_output[0]
        
        # Para usar o AdaptiveAvgPool1d, precisamos que a dimensão a ser reduzida (sequence_length)
        # seja a última. O shape atual é [batch, seq_len, hidden_size].
        # Precisamos trocá-lo para [batch, hidden_size, seq_len].
        permuted_state = last_hidden_state.permute(0, 2, 1)
        
        # Aplica o pooling na dimensão da sequência.
        # Shape da saída: [batch_size, hidden_size, 1]
        pooled_output = self.pooling(permuted_state)
        
        # Remove a última dimensão que agora é 1.
        # Shape da saída: [batch_size, hidden_size]
        squeezed_output = pooled_output.squeeze(-1)
        
        # Passa o vetor de características pela camada linear para obter os logits da classificação.
        linear_output = self.fc1(squeezed_output)
        
        return linear_output
