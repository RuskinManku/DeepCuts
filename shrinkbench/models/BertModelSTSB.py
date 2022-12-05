import torch.nn as nn
from transformers import BertModel,BertTokenizer
class BertNet(nn.Module):
    def __init__(self, pretrained=False):
        # assert not pretrained, f"{self.__class__.__name__} does not support pretrained weights"
        super(BertNet, self).__init__()
        self.bert_model=BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
        self.regressor=nn.Linear(768,1)
        setattr(self.regressor, "is_classifier", True)

    def forward(self, x):
        x_0=list(x[0])
        x_1=list(x[1])
        tokenized_out=self.tokenizer(x_0,x_1,padding=True,return_tensors="pt").to("cuda:0")
        bert_out=self.bert_model(**tokenized_out).pooler_output
        return 5*nn.functional.sigmoid(self.regressor(bert_out))