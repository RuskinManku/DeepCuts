import torch.nn as nn
from transformers import BertModel,BertTokenizer
class BertNet(nn.Module):
    def __init__(self, pretrained=False):
        # assert not pretrained, f"{self.__class__.__name__} does not support pretrained weights"
        super(BertNet, self).__init__()
        self.bert_model=BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
        self.classifier=nn.Linear(768,2)
        setattr(self.classifier, "is_classifier", True)

    def forward(self, x):
        tokenized_out=self.tokenizer(x,padding=True,return_tensors="pt").to("cuda:0")
        bert_out=self.bert_model(**tokenized_out).pooler_output
        return self.classifier(bert_out)