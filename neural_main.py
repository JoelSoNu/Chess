#!/bin/env python3

import torch
from torch import nn
import random

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

def fill_memory(env, agent, memory_fill_eps):
    for _ in range(memory_fill_eps):
        state = env.getCurrentBoard()
        done = False
        while not done:
            #White (the agent)
            action = agent.select_action(state)
            env.makeMove(action)
            next_state = env.getCurrentBoard
            done = env.inCheckMate() or env.itsDraw()
            agent.replay_memory.store(state, action, next_state, done)
            state = next_state

            #Black random
            moves, movesID = env.getValidMoves()
            move = moves[random.randint(0, len(moves) - 1)]
            env.makeMove(move)
            next_state = env.getCurrentBoard
            done = env.inCheckMate() or env.itsDraw()
            agent.replay_memory.store(state, action, next_state, done)
            state = next_state
        if done:
            if env.itsDraw():
                reward = 500
            elif env.inCheckMate() and env.blackToMove():
                reward = 1000
            else:
                reward = 0
            agent.replay_memory.store_rewards(reward, 0.99)

def train(env, agent, train_eps, memory_fill_eps, batchsize, update_freq, model_filename):
    fill_memory(env, agent, memory_fill_eps)
    print("Samples in memory: ", len(agent.replay_memory))

    step_cnt = 0
    reward_history = []

def main():
    gameState = engine.GameState()
    policyNet = nr.ChessNet(gameState, args)
    targetNet = nr.ChessNet(gameState, args)
    sgd = torch.optim.SGD(policyNet.parameters(), lr=0.1)
    loss = nn.MSELoss()
    netContext = nr.NetContext(gameState, args, 0.99, 0.01, 1.0, 0.95, 10000, loss)   #  epsilon max = 1.0 , epsilon min = 0.01
    board = gameState.boardAsNumbers()
    print(netContext.convertToTensor(board))
    print(board)
    print(len(board))
    with torch.no_grad():
        qValues = netContext.getQValues(netContext.targetNet)
    print(qValues)
    netContext.train(2, 2)
    print(board == gameState.boardAsNumbers())
    with torch.no_grad():
        qValues = netContext.getQValues(netContext.targetNet)
    print(qValues)
    netContext.train(2, 2)
    print(board == gameState.boardAsNumbers())
    with torch.no_grad():
        qValues = netContext.getQValues(netContext.targetNet)
    print(qValues)
    netContext.train(2, 2)
    print(board == gameState.boardAsNumbers())
    with torch.no_grad():
        qValues = netContext.getQValues(netContext.targetNet)
    print(qValues)

if __name__ == "__main__":
    main()