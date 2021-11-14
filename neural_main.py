#!/bin/env python3

import torch
from torch import nn

import engine
import neural_reinforcement as nr

args = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
}

def main():
    gameState = engine.GameState()
    policyNet = nr.ChessNet(gameState, args)
    targetNet = nr.ChessNet(gameState, args)
    sgd = torch.optim.SGD(policyNet.parameters(), lr=0.1)
    loss = nn.MSELoss()
    netContext = nr.NetContext(gameState, policyNet, targetNet, sgd, loss)
    print(netContext.convertToTensor(gameState.boardAsNumbers()))
    print(gameState.boardAsNumbers())
    print(len(gameState.boardAsNumbers()))
    with torch.no_grad():
        qValues = netContext.getQValues(netContext.targetNet)
    print(qValues)

if __name__ == "__main__":
    main()