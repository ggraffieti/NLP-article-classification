from torch.utils.data import Dataset
import torch


class ArticleDataset(Dataset):
    def __init__(self, title_encodings, text_encodings, titletext_encodings, labels):
        super(ArticleDataset, self).__init__()
        self.title_encodings = title_encodings
        self.text_encodings = text_encodings
        self.titletext_encodings = titletext_encodings
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        return self.labels[idx], self.title_encodings[idx], self.text_encodings[idx], self.titletext_encodings[idx]

    def __len__(self):
        return len(self.labels)
