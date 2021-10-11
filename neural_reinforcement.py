#!/bin/env python3

import torch
from torch import nn

WIN_VALUE = 1.0
DRAW_VALUE = 0.8
LOSS_VALUE = 0.0

INPUT_SIZE = 2  # board and moves
OUTPUT_SIZE = 1  # dict (moves: q_value)


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(INPUT_SIZE, 36)
        self.hidden_layer = nn.Linear(36, 36)
        self.hidden_layer_2 = nn.Linear(36, 36)
        self.output_layer = nn.Linear(36, OUTPUT_SIZE)

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.relu(x)

        x = self.hidden_layer(x)
        x = torch.relu(x)

        x = self.hidden_layer_2(x)
        x = torch.relu(x)

        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

class NetContext():
    def __init__(self, policyNet, targetNet, optimizer, lossFunction):
        self.policyNet = policyNet

        self.targetNet = targetNet
        self.targetNet.load_state_dict(self.policyNet.state_dict())
        self.targetNet = self.targetNet.eval()

        self.optimizer = optimizer
        self.lossFunction = lossFunction

    def playTrainingGames(self):
        pass

    def updateTrainingGameover(self):
        pass

    def backpropagate(self):
        pass