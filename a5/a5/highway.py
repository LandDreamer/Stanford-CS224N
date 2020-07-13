#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, word_embed_size):
        super(Highway, self).__init__()
        self.word_embed_size = word_embed_size
        self.proj = nn.Linear(self.word_embed_size, self.word_embed_size, bias=True)
        self.gate = nn.Linear(self.word_embed_size, self.word_embed_size, bias=True)


    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """
        @param x_conv_out(torch.Tensor): shape(batch_size, word_embed_size)
        
        @return x_highway(torch.Tensor): shape(batch_size, word_embed_size)
        """
        x_proj = self.proj(x_conv_out)
        x_proj = F.relu(x_proj)
        x_gate = self.gate(x_conv_out)
        x_gate = torch.sigmoid(x_gate)

        x_highway = x_gate*x_proj + (1-x_gate)*x_conv_out

        return x_highway

    ### END YOUR CODE

