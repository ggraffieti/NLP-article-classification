import torch.nn as nn
from transformers import BertForSequenceClassification


class Bert(nn.Module):
    def __init__(self, bert_model):
        super(Bert, self).__init__()
        self.encoder = BertForSequenceClassification.from_pretrained(bert_model)

    def forward(self, text, label):
        loss, test_fea = self.encoder(text, labels=label)[:2]
        return loss, test_fea
