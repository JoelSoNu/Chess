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

    def getQValues(self, gameState, model):
        inputs = self.convertToTensor(gameState)
        outputs = model(inputs)
        return outputs

    def convertToTensor(self, gameState):
        pass
        #use moves and convert strings to numbers? a2a4 can be [?, ?, ?, ?]
        #return torch.tensor(gameState.board, dtype=torch.)

    def playTrainingGames(self):
        # play game
        # updateTrainingGameover
        pass

    def updateTrainingGameover(self):
        # calls backpropagate
        pass

    def backpropagate(self):
        # net_context.optimizer.zero_grad()
        # loss.backward()
        # net_context.optimizer.step()
        pass