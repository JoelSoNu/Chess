#!/bin/env python3
import os

import numpy as np
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
        state = env.boardAsNumbers()
        done = False
        while not done:
            #White (the agent)
            action = agent.select_action(state)
            moves, movesID = env.getValidMoves()
            env.makeMove(moves[action])
            next_state = env.boardAsNumbers()
            done = env.inCheckMate() or env.itsDraw()
            agent.replay_memory.store(state, action, next_state, done)
            state = next_state
            if done:
                break
            #Black random
            moves, movesID = env.getValidMoves()
            move = moves[random.randint(0, len(moves) - 1)]
            env.makeMove(move)
            next_state = env.boardAsNumbers()
            done = env.inCheckMate() or env.itsDraw()
            agent.replay_memory.store(state, action, next_state, done)
            state = next_state
            if done:
                break
        if done:
            if env.itsDraw():
                reward = 500
            elif env.inCheckMate() and not env.whiteToMove:
                reward = 1000
            else:
                reward = 0
            agent.replay_memory.store_rewards(reward, 0.99)

def train(env, agent, train_eps, memory_fill_eps, batchsize, update_freq, model_filename):
    fill_memory(env, agent, memory_fill_eps)
    print("Samples in memory: ", len(agent.replay_memory))

    step_cnt = 0
    reward_history = []
    best_score = -np.inf

    for ep_cnt in range(train_eps):
        state = env.boardAsNumbers()
        done = False
        ep_reward = 0

        while not done:
            # White (the agent)
            action = agent.select_action(state)
            moves, movesID = env.getValidMoves()
            env.makeMove(moves[action])
            next_state = env.boardAsNumbers()
            done = env.inCheckMate() or env.itsDraw()
            agent.replay_memory.store(state, action, next_state, done)
            state = next_state
            step_cnt = step_cnt + 1
            if done:
                break
            # Black random
            moves, movesID = env.getValidMoves()
            move = moves[random.randint(0, len(moves) - 1)]
            env.makeMove(move)
            next_state = env.boardAsNumbers()
            done = env.inCheckMate() or env.itsDraw()
            agent.replay_memory.store(state, action, next_state, done)
            state = next_state

            if step_cnt % update_freq == 0:
                agent.update_target()
            if done:
                break
        if done:
            if env.itsDraw():
                ep_reward = 500
            elif env.inCheckMate() and env.blackToMove():
                ep_reward = 1000
            else:
                ep_reward = 0
        agent.replay_memory.store_rewards(ep_reward, 0.99)
        agent.update_epsilon()
        reward_history.append(ep_reward)
        current_avg_store = np.mean(reward_history[-100:])

        print('Ep: {}, Total Steps: {}, Ep Score: {}, Avg Score: {}, Updated Epsilon: {}'.format(
            ep_cnt, step_cnt, ep_reward, current_avg_store, agent.epsilon))

        if current_avg_store >= best_score:
            agent.save(model_filename)
            best_score = current_avg_store

def test(env, agent, test_eps):
    for ep_cnt in range(test_eps):
        state = env.boardAsNumbers()
        done = False
        ep_reward = 0

        while not done:
            # White (the agent)
            action = agent.select_action(state)
            moves, movesID = env.getValidMoves()
            env.makeMove(moves[action])
            next_state = env.boardAsNumbers()
            done = env.inCheckMate() or env.itsDraw()
            state = next_state
            if done:
                break
            # Black random
            moves, movesID = env.getValidMoves()
            move = moves[random.randint(0, len(moves) - 1)]
            env.makeMove(move)
            next_state = env.boardAsNumbers()
            done = env.inCheckMate() or env.itsDraw()
            state = next_state
            if done:
                break
        if done:
            if env.itsDraw():
                ep_reward = 500
            elif env.inCheckMate() and not env.whiteToMove:
                ep_reward = 1000
            else:
                ep_reward = 0

        print('Ep: {}, Ep Score: {}'.format(ep_cnt, ep_reward))


def main():
    train_mode = True
    env = engine.GameState()
    model_filename = "AlphaZero"
    loss = nn.MSELoss()
    if train_mode:
        agent = nr.NetContext(env, args, 0.99, 0.01, 1.0, 0.95, 1000000, loss)
        train(env=env, agent=agent, train_eps=200, memory_fill_eps=20, batchsize=64, update_freq=100, model_filename=model_filename)
    else:
        agent = nr.NetContext(env, args, 0.99, 0.0, 0.0, 0.0, 1000000, loss)
        agent.load(model_filename)

        test(env=env, agent=agent, test_eps=100)


if __name__ == "__main__":
    main()