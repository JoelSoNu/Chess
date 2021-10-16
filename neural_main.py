#!/bin/env python3

import torch
from torch import nn

import engine
import neural_reinforcement as nr

def main():
    policyNet = nr.ChessNet()
    targetNet = nr.ChessNet()
    sgd = torch.optim.SGD(policyNet.parameters(), lr=0.1)
    loss = nn.MSELoss()
    netContext = nr.NetContext(policyNet, targetNet, sgd, loss)
    gameState = engine.GameState()
    print(netContext.convertToTensor(gameState))
    with torch.no_grad():
        qValues = netContext.getQValues(gameState, netContext.targetNet)
    print(qValues)

if __name__ == "__main__":
    main()