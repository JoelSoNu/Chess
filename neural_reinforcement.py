#!/bin/env python3

import sys
from collections import namedtuple, deque
import random
from utils import *
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


WIN_VALUE = 1.0
DRAW_VALUE = 0.8
LOSS_VALUE = 0.0

INPUT_SIZE = 2  # board and moves
OUTPUT_SIZE = 1  # dict (moves: q_value)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ChessNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args.get("num_channels"), (3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(args.get("num_channels"), args.get("num_channels"), (3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(args.get("num_channels"), args.get("num_channels"), (3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(args.get("num_channels"), args.get("num_channels"), (3, 3), stride=(1, 1))

        self.bn1 = nn.BatchNorm2d(args.get("num_channels"))
        self.bn2 = nn.BatchNorm2d(args.get("num_channels"))
        self.bn3 = nn.BatchNorm2d(args.get("num_channels"))
        self.bn4 = nn.BatchNorm2d(args.get("num_channels"))

        self.fc1 = nn.Linear(args.get("num_channels")*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.args.get("num_channels")*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.get("dropout"), training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.get("dropout"), training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def updateActionSize(self, size):
        self.action_size = size


class NetContext():
    def __init__(self, gameState, policyNet, targetNet, optimizer, lossFunction):
        self.board_x, self.board_y = gameState.getBoardSize()
        self.actionSize = gameState.getActionSize()
        self.gameState = gameState

        self.policyNet = policyNet

        self.targetNet = targetNet
        self.targetNet.load_state_dict(self.policyNet.state_dict())
        self.targetNet = self.targetNet.eval()

        self.optimizer = optimizer
        self.lossFunction = lossFunction

    def getQValues(self, model):
        inputs = self.convertToTensor(self.gameState.boardAsNumbers())
        outputs = model(inputs)
        return outputs

    def convertToTensor(self, board):
        return torch.tensor(board, dtype=torch.float)

    def train(self, examples, args):
        moves, movesID = self.gameState.getValidMoves()
        print(len(moves))
        self.gameState.makeMove(moves[0])
        moves2, movesID2 = self.gameState.getValidMoves()
        self.targetNet.updateActionSize(len(moves2))
        '''for epoch in range(args.get("epochs")):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.get("batch_size")'''

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