import csv
import string
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# #setting the device to "cuda" if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import __main__

class Transformer_model(nn.Module):
    def __init__(self, embed_size, len_src_vocab, len_tgt_vocab, src_pad_index, num_heads, enc_layers, dec_layers, forward_exp, dropout, max_length,device):
        super(Transformer_model,self).__init__()
        self.src_word_embedding=nn.Embedding(len_src_vocab, embed_size) #shape: (len_src_vocab, embed_size)
        self.tgt_word_embedding=nn.Embedding(len_tgt_vocab, embed_size) #shape: (len_eng_vocab, embed_size)
        self.src_positional_embedding=nn.Embedding(max_length, embed_size) #shape: (MAX_LENGTH, embed_size)
        self.tgt_positional_embedding=nn.Embedding(max_length, embed_size)  #shape: (MAX_LENGTH, embed_size)
        self.device=device
        self.transformer_layer=nn.Transformer(embed_size, num_heads,enc_layers, dec_layers, forward_expansion, dropout)
        self.out_fc=nn.Linear(embed_size, len_tgt_vocab)    #linear layer to predicted the output word
        self.dropout=nn.Dropout(dropout)
        self.src_pad_index=src_pad_index

    def gen_mask_for_src(self, source):
        #need to transpose source as padding need to be of size (batch_size, seq_len) but source is of shape (seq_len, batch_size)
        source=source.transpose(0,1)
        mask=(source==self.src_pad_index) #(mask will contain 1 where there is pad token, and 0 otherwise)
        return mask.to(self.device)

    def forward(self, src, target):
        src_seq_length, batch_size=src.shape
        tgt_seq_length, batch_size=target.shape
        # creating positional embeddings to encode position of words in transformer (it will be just a range array upto max_length)
        src_positional=torch.arange(0,src_seq_length).unsqueeze(1).expand(src_seq_length, batch_size).to(self.device)
        tgt_positional=torch.arange(0,tgt_seq_length).unsqueeze(1).expand(tgt_seq_length, batch_size).to(self.device)
        # calculating embeddings as sum of positional and word embeddings
        src_embedding=self.dropout(self.src_word_embedding(src)+self.src_positional_embedding(src_positional))
        tht_embedding=self.dropout(self.tgt_word_embedding(target)+self.tgt_positional_embedding(tgt_positional))
        # generating padding mask for hindi (source)
        src_padding_mask=self.gen_mask_for_src(src)
        # using in-built transformer function to generate mask for english (target)
        # It will be in form of a lower-triangular matrix
        tgt_mask=self.transformer_layer.generate_square_subsequent_mask(tgt_seq_length).to(self.device)
        output=self.transformer_layer(src_embedding, tht_embedding, src_key_padding_mask=src_padding_mask, tgt_mask=tgt_mask)
        output=self.out_fc(output)
        return output



__main__.Transformer_model = Transformer_model




