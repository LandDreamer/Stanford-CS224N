#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)
        self.loss_func = nn.CrossEntropyLoss(
                            reduction = 'sum',
                            ignore_index = target_vocab.char_pad
                        )

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.
        # a word has 'length' charaters

        x = self.decoderCharEmb(input) # (length, batch_size, char_embed_size)
        # if not dec_hidden:
        #     h0 = torch.zeros((1, input.size(1), self.hidden_size))
        #     c0 = torch.zeros((1, input.size(1), self.hidden_size))
        #     dec_hidden = (h0, c0)

        dec_hiddens, (last_hidden, last_cell) = self.charDecoder(x, dec_hidden)
        # dec_hiddens (length, batch_size, hidden_size)
        # last_hidden/last_cell (1, batch, hidden_size)

        s = self.char_output_projection(dec_hiddens) # (length, batch_size, self.vocab_size)

        return s, (last_hidden, last_cell)
        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss

        input = char_sequence[:-1] # (length-1, batch_size)
        s, (last_hidden, lats_cell) = self.forward(input, dec_hidden=dec_hidden)
        # s (length-1, batch_size, self.vocab_size)
        p = s # (length-1, batch_size, self.vocab_size)
        x = char_sequence[1:] # (length-1, batch_size)

        loss = 0.
        for p_t, x_t in zip(p,x):
            loss += self.loss_func(p_t, x_t) # nn.CrossEntropyLoss will do softmax

        return loss
        ### END YOUR CODE



    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size = initialStates[0].size(1)
        output_words = []

        for i in range(batch_size):
            char = self.target_vocab.start_of_word
            word = []
            dec_hidden = (initialStates[0][:,i:i+1,:], initialStates[1][:,i:i+1,:])
            # tuple contains 2 (1, 1, hidden_size)
            for t in range(max_length):
                input = torch.tensor([char], device=device).unsqueeze(dim=0)
                s, dec_hidden = self.forward(input, dec_hidden=dec_hidden)
                p = F.softmax(s,dim=2) # (1, 1, char_vocab_size)
                char = torch.argmax(p)   
                if char == self.target_vocab.end_of_word:
                    break
                word.append(char.item())

            word = [self.target_vocab.id2char[c] for c in word]
            word = "".join(word)
            output_words.append(word)

        return output_words

        ### END YOUR CODE

