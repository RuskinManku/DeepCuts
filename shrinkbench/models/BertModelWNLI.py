import torch.nn as nn
from transformers import BertModel,BertTokenizer
class BertModelWNLI(nn.Module):
    def __init__(self, pretrained=False):
        # assert not pretrained, f"{self.__class__.__name__} does not support pretrained weights"
        super(BertModelWNLI, self).__init__()
        self.bert_model=BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
        self.classifier=nn.Linear(768,2)
        setattr(self.classifier, "is_classifier", True)

    def forward(self, x):
        x_0=list(x[0])
        x_1=list(x[1])
        tokenized_out=self.tokenizer(x_0,x_1,padding=True,return_tensors="pt").to("cuda:0")
        bert_out=self.bert_model(**tokenized_out).pooler_output
        return self.classifier(bert_out)