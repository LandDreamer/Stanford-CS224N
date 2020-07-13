#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    ### YOUR CODE HERE for part 1g
    def __init__(self, filter_num, char_embed_size, m_word=21, kernel_size=5, padding_num=1):
        super(CNN, self).__init__()

        self.m_word = m_word
        self.char_embed_size = char_embed_size
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.padding_num = padding_num
        self.conv = nn.Conv1d(in_channels=self.char_embed_size, out_channels=self.filter_num, 
                            kernel_size=self.kernel_size, padding=self.padding_num)


    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        """
        @param x_reshaped(torch.Tensor): shape (batch_size, char_embed_size, m_word)

        @return x_conv_out(torch.Tensor): shape (batch_size, filter_num(word_embed_size))
        """

        x_temp = self.conv(x_reshaped)
        x_temp = F.relu(x_temp) #shape(batch_size, filter_num(word_embed_size), ***)
        self.maxpool = nn.MaxPool1d(kernel_size=x_temp.size(2))
        x_conv_out = self.maxpool(x_temp)
        x_conv_out = torch.squeeze(x_conv_out, dim=2)

        return x_conv_out

    ### END YOUR CODE

