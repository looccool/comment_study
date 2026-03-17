import os, sys, time, numpy, pandas
import torch
import torch.nn as nn
import torch.nn.functional as F


class toxEncoderBlock(nn.Module):
    def __init__(self, embedDim, numHeads, dropout, batchFirst, FFDim):
        super().__init__()

        # for attention:
        self.attention = nn.MultiheadAttention(embed_dim=embedDim,
                                               num_heads=numHeads,
                                               dropout=dropout,
                                               batch_first=batchFirst,
                                               )

        # for feed-forward net:
        self.linearEmb2FF = nn.Linear(embedDim, FFDim)
        self.dropout = nn.Dropout(dropout)
        self.linearFF2Emb = nn.Linear(FFDim, embedDim)

        # for layer-normalization:
        self.normAttn = nn.LayerNorm(embedDim)
        self.normFF = nn.LayerNorm(embedDim)
        self.dropoutAttn = nn.Dropout(dropout)
        self.dropoutFF = nn.Dropout(dropout)

    def forward(self, x, paddingMask):
        # pre-layer-normalization before attention:
        xNormAttn = self.normAttn(x)

        # apply multi-head attention:
        attnOutput, attnWeights = self.attention(
            query = xNormAttn,
            key = xNormAttn,
            value = xNormAttn,
            key_padding_mask = paddingMask
        )

        # skip-connection:
        xDropAttn = self.dropoutAttn(attnOutput)
        x = x + xDropAttn

        # pre-layer-normalization for feed-forward layer:
        xNormFF = self.normFF(x)
        # feed-forward network:
        FFOutput = self.linearFF2Emb( self.dropout( F.relu( self.linearEmb2FF(xNormFF) ) ) )
        # skip-connection:
        xDropFF = self.dropoutFF(FFOutput)
        x = x + xDropFF

        # return:
        return x
##########################################################
##########################################################
class toxCLFModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # token embedding:
        self.embedding = nn.Embedding(num_embeddings=self.config['vocabSize'],
                                      embedding_dim=self.config['embeddingDim']
                                      )
        # positional encoding:
        self.posEncodeing = nn.Parameter(torch.randn(1, self.config['windowSize'], self.config['embeddingDim']))

        # encoder block layers:
        self.encLayers = nn.ModuleList([
            toxEncoderBlock(embedDim=self.config['embeddingDim'],
                            numHeads=self.config['numHeads'],
                            dropout=self.config['dropout'],
                            batchFirst=self.config['batchFirst'],
                            FFDim=self.config['FFDim'])
            for _ in range(self.config['numLayers'])
        ])

        # layer-norm before classification head:
        self.LNPreCLF = nn.LayerNorm(self.config['embeddingDim'])
        self.getLogits = nn.Linear(self.config['embeddingDim'], self.config['numClasses'])

    def forward(self, x, paddingMask):
        batchSize, SeqLen = x.shape
        # embedding:
        x = self.embedding(x)
        # add positional encoding
        x = x + self.posEncodeing[:, 0: SeqLen, :]
        # go through transformer:
        for block in self.encLayers:
            x = block(x, paddingMask)
        # layer-norm:
        x = self.LNPreCLF(x)
        # use mean of the final embeddings as input for classification:
        if paddingMask==None:
            x = x.mean(dim=1)
        else:
            mask = (~paddingMask).float().unsqueeze(-1)
            x = x * mask
            x = x.sum(dim=1) / mask.sum(dim=1)
        # logits:
        logits = self.getLogits(x)
        # probs:
        probs = torch.softmax(logits, dim=-1)

        return logits, probs