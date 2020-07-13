#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab, dropout_rate=0.3, char_embed_size=50):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        self.max_word_length = 21
        self.word_embed_size = word_embed_size
        self.vocab = vocab
        self.dropout = nn.Dropout(dropout_rate)
        self.char_embed_size = char_embed_size

        src_pad_token_idx = vocab.char2id['<pad>']

        self.source = nn.Embedding(len(self.vocab.char2id), self.char_embed_size, 
                                    padding_idx=src_pad_token_idx)

        self.highway = Highway(self.word_embed_size)
        self.cnn = CNN(filter_num=self.word_embed_size, char_embed_size=self.char_embed_size)

        ### END YOUR CODE


    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        # print("word_size %d" %self.word_embed_size)
        # print("char_size %d" %self.char_embed_size)
        # print("batch_size %d" %input.size(1))
        # print("sentence_len %d" %input.size(0))

        ### YOUR CODE HERE for part 1h
        x_emb = self.source(input) # (sentence_length, batch_size, max_word_length, char_embed_size)
        x_reshape = x_emb.transpose(2,3) # (sentence_length, batch_size, char_embed_size, max_word_length)
        
        x_conv_out = []
        for sent in x_reshape: # sent: (batch_size, char_embed_size, max_word_length)
            x_temp = self.cnn(sent)
            x_conv_out.append(x_temp)
        x_conv_out = torch.stack(x_conv_out, dim=0)  # (sentence_length, batch_size, word_embed_size)

        x_highway = []
        for sent in x_conv_out:
            x_temp = self.highway(sent)
            x_highway.append(x_temp)
        x_highway = torch.stack(x_highway, dim=0)

        x_word_emb = self.dropout(x_highway)

        return x_word_emb
        ### END YOUR CODE

