import csv
from collections import Counter
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torchtext

def load_data(path):
    texts = []
    
    with open(path, 'r') as f:
        data_Reader = csv.reader(f)
        for text in tqdm(data_Reader):
            texts.append(text)
    
    return texts

def build_vocab(texts):
    counter = Counter()
    
    texts = texts
    
    for text in tqdm(texts):
        counter.update(text)
        
    return torchtext.vocab.vocab(counter, specials=['<unk>', '<pad>', '<start>', '<end>'])

def convert_text_to_indexes(text, vocab):
    return [vocab['<start>']] + [vocab[token] for token in text] + [vocab['<end>']]

def data_process(texts_src, texts_tgt, vocab_src, vocab_tgt):
    data = []
    
    for src, tgt in tqdm(zip(texts_src, texts_tgt)):
        src_tensor = torch.tensor(convert_text_to_indexes(text=src, vocab=vocab_src), dtype=torch.long)
        tgt_tensor = torch.tensor(convert_text_to_indexes(text=tgt, vocab=vocab_tgt), dtype=torch.long)
        data.append((src_tensor, tgt_tensor))
    return data

def batch_generator(data_batch):
    batch_src, batch_tgt = [], []
    
    for src, tgt in data_batch:
        batch_src.append(src)
        batch_tgt.append(tgt)
    
    batch_src = nn.utils.rnn.pad_sequence(batch_src, padding_value=1)
    batch_tgt = nn.utils.rnn.pad_sequence(batch_tgt, padding_value=1)
    
    return batch_src, batch_tgt

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size
    
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_size)
    
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout, maxlen=10000):
        super(PositionalEncoding, self).__init__()
        
        den = torch.exp(-torch.arange(0, embedding_size, 2) * math.log(100000) / embedding_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        
        embedding_pos = torch.zeros((maxlen, embedding_size))
        embedding_pos[:, 0::2] = torch.sin(pos * den)
        embedding_pos[:, 1::2] = torch.cos(pos * den)
        embedding_pos = embedding_pos.unsqueeze(-2)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('embedding_pos', embedding_pos)
    
    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.embedding_pos[: token_embedding.size(0), :])

def generate_square_subsequent_mask(seq_len, PAD_IDX):
    mask = (torch.triu(torch.ones((seq_len, seq_len))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == PAD_IDX, float(0.0))

    return mask

def create_mask(src, tgt, PAD_IDX):
    seq_len_src = src.shape[0]
    seq_len_tgt = tgt.shape[0]
    
    mask_tgt = generate_square_subsequent_mask(seq_len_tgt, PAD_IDX)
    mask_src = torch.zeros((seq_len_src, seq_len_src)).type(torch.bool)
    
    padding_mask_src = (src == PAD_IDX).transpose(0, 1)
    padding_mask_tgt = (tgt == PAD_IDX).transpose(0, 1)
    
    return mask_src, mask_tgt, padding_mask_src, padding_mask_tgt

