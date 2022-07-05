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


#DQN Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.idx = 0

    def store(self, states, actions, next_states, rewards, dones):
        if len(self.states) < self.capacity:
            self.states.append(states)
            self.actions.append(actions)
            self.next_states.append(next_states)
            self.rewards.append(rewards)
            self.dones.append(dones)
        else:
            self.states[self.idx] = states
            self.actions[self.idx] = actions
            self.next_states[self.idx] = next_states
            self.rewards[self.idx] = rewards
            self.dones[self.idx] = dones

        self.idx = (self.idx + 1) % self.capacity


    def sample(self, batch_size, device):
        indices_to_sample = random.sample(range(len(self.states)), k=batch_size)
        states = torch.from_numpy(np.array(self.states)[indices_to_sample]).float().to(device)
        actions = torch.from_numpy(np.array(self.actions)[indices_to_sample]).float().to(device)
        next_states = torch.from_numpy(np.array(self.next_states)[indices_to_sample]).float().to(device)
        rewards = torch.from_numpy(np.array(self.rewards)[indices_to_sample]).float().to(device)
        dones = torch.from_numpy(np.array(self.dones)[indices_to_sample]).float().to(device)

        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.states)


#DQN Net
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

        self.fc1 = nn.Linear(args.get("num_channels") * self.board_x * self.board_y, 1024)
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


#DQN Agent
class NetContext():
    def __init__(self, gameState, args, epsilon_min, epsilon_max, lossFunction):
        self.board_x, self.board_y = gameState.getBoardSize()
        self.actionSize = gameState.getActionSize()
        self.gameState = gameState
        self.device = torch.device("cuda" if args.get("cuda") else "cpu")

        self.policyNet = ChessNet(gameState, args).to(self.device)

        self.targetNet = ChessNet(gameState, args).to(self.device)
        self.targetNet.load_state_dict(self.policyNet.state_dict())
        self.targetNet = self.targetNet.eval()
        self.targetNet.load_state_dict(self.policyNet.state_dict())

        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.optimizer = torch.optim.SGD(self.policyNet.parameters(), lr=0.1)
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