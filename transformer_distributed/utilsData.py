import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


def getTokenListSize(text, tokenizer):
    token_ids = tokenizer.encode(text)
    token_bytes_list = [tokenizer.decode_single_token_bytes(token) for token in token_ids]
    token_strings = [token.decode("utf-8", errors="replace") for token in token_bytes_list]
    return len(token_strings)

def getMaxWindow(df, commentColumn, tokenizer):
    tmp = df[commentColumn].apply(lambda x: getTokenListSize(x.strip(), tokenizer))
    return max(tmp)

class datasetForTox(Dataset):
    def __init__(self, dataFile, maxWindowSize, dataConfig):
        self.config = dataConfig
        # load data into memory:
        self.df = pd.read_csv(dataFile, sep=self.config['separator'])
        # reset index:
        self.df = self.df.reset_index(drop=False)
        # create binary label:
        self.df['label'] = self.df[self.config['toxColumn']].apply(lambda x: 1 if x >= self.config['toxThreshold'] else 0)
        # create tokenizer:
        self.tokenizer = tiktoken.get_encoding('gpt2')
        # get max window size:
        self.maxWin = maxWindowSize
        if self.maxWin == None:
            self.maxWin = getMaxWindow(df=self.df, commentColumn=self.config['commentColumn'], tokenizer=self.tokenizer)
        # get pad_id:
        self.pad_id = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        comment = self.df.loc[idx, self.config['commentColumn']]
        token_ids = self.tokenizer.encode(comment.strip())

        num_pads = self.maxWin - len(token_ids)
        if num_pads > 0:
            padded_seq = token_ids + [self.pad_id] * num_pads
            mask = [False] * len(token_ids) + [True] * num_pads
        else:
            padded_seq = token_ids[0: self.maxWin]
            mask = [False]* self.maxWin

        x = torch.tensor(padded_seq, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.bool)

        y = self.df.loc[idx, 'label']
        y = torch.tensor(y, dtype=torch.long)

        return x, mask, y