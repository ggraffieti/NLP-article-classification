from transformers import BertTokenizer
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from data.article_dataset import ArticleDataset
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader


def get_train_valid_test(bert_model, max_seq_lenght, device):
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    unk_index = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # Fields
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                       fix_length=max_seq_lenght, pad_token=pad_index, unk_token=unk_index)
    fields = [('label', label_field), ('title', text_field), ('text', text_field), ('titletext', text_field)]

    # TabularDataset
    train, valid, test = TabularDataset.splits(path="../data", train='train.csv', validation='valid.csv',
                                               test='test.csv', format='CSV', fields=fields, skip_header=True)

    # Iterators
    train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    test_iter = Iterator(test, batch_size=16, device=device, train=False, shuffle=False, sort=False)

    return train_iter, valid_iter, test_iter


def _get_dataloader(file_path, tokenizer, max_seq_lenght, drop_last=True):
    train_csv = pd.read_csv(file_path)

    titles = train_csv["title"].tolist()
    text = train_csv["text"].tolist()
    titletext = train_csv["titletext"].tolist()
    labels = train_csv["label"]

    titles_enc = tokenizer(titles, truncation=True, max_length=max_seq_lenght, padding='max_length',
                           return_tensors="pt")
    text_enc = tokenizer(text, truncation=True, max_length=max_seq_lenght, padding='max_length', return_tensors="pt")
    titletext_enc = tokenizer(titletext, truncation=True, max_length=max_seq_lenght, padding='max_length',
                              return_tensors="pt")

    article_dataset = ArticleDataset(titles_enc.input_ids, text_enc.input_ids, titletext_enc.input_ids, labels)

    return DataLoader(article_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last)


def get_train_valid_test_new(bert_model, max_seq_lenght):
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    train_dl = _get_dataloader("../data/train.csv", tokenizer, max_seq_lenght)
    valid_dl = _get_dataloader("../data/valid.csv", tokenizer, max_seq_lenght, drop_last=False)
    test_dl = _get_dataloader("../data/test.csv", tokenizer, max_seq_lenght, drop_last=False)

    return train_dl, valid_dl, test_dl

#
# if __name__ == "__main__":
#     dl = get_train_valid_test_new("bert-base-uncased", 128, "cpu")
#     for lab, t, tx, ttx in dl:
#         print(lab)
#         print(t)
#         print(tx)
#         print(ttx)
#         break
