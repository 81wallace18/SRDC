import torch
from torch import nn 
from transformers import BertModel, BertTokenizer # Alterado para BERT

class Classifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, bert_model_name:str, compression_ratio:int):
        super(Classifier,self).__init__()
        # Carrega o modelo BERT
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.pooling = nn.AdaptiveAvgPool1d(compression_ratio)
        
        # O BERT Base também tem hidden_size de 768, similar ao GPT-2 Small.
        # Se usar BERT Large, mude para 1024.
        self.fc1 = nn.Linear(compression_ratio*max_seq_len*7, num_classes)
       
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
            # return_dict=False retorna (last_hidden_state, pooler_output)
            # Nós queremos o last_hidden_state (bert_out)
            bert_out, _ = self.bert_model(input_ids=sub_input_id, attention_mask=sub_mask, return_dict=False)
            
            bert_out_pooling = self.pooling(bert_out)
            batch_size = bert_out_pooling.shape[0]
            concatenated_sub_tensors.append(bert_out_pooling)
       
        result = torch.cat(concatenated_sub_tensors, dim=1)
        batch_size = result.shape[0]
        linear_output = self.fc1(result.view(batch_size,-1))
        return linear_output