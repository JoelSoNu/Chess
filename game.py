#!/bin/env python3

import pygame
import time
import sys
import array
import math
import random

#from PIL import Image
#import matplotlib.image as img

import engine
import minimax_abPrunning as minmax
import neural_reinforcement as nr
import torch
from torch import nn


WIDTH = HEIGHT = 512
DIMENSION = 8  #dimension of chess board is 8x8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 30
IMAGES = {}
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
playerWhite = "random"
playerBlack = "alpha"

args = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
}

def loadImages():
    pieces = ['wp', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bp', 'bN', 'bB', 'bR', 'bQ', 'bK']
    for piece in pieces:
        IMAGES[piece] = pygame.transform.scale(pygame.image.load("images/" + piece + ".png"), (SQ_SIZE, SQ_SIZE))
    #we can acces an image by saying 'IMAGES['wp']


def main():
    pygame.init()
    #screen = pygame.display.set_mode((WIDTH, HEIGHT))
    #clock = pygame.time.Clock()
    screen.fill(pygame.Color('white'))
    gs = engine.GameState()
    loadImages()
    pygame.display.set_caption("Chess board")
    running = True
    stopGame = False
    #----
    model_filename = "AlphaZero"
    loss = nn.MSELoss()
    agent = nr.NetContext(gs, args, 0.99, 0.0, 0.0, 0.0, 100000, loss)
    agent.load(model_filename)
    #----
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    gs.goBackMove()
                elif event.key == pygame.K_RIGHT:
                    gs.goForthMove()
            elif gs.inCheckMate():  # Checkmate
                winner = "BLACK WINS" if gs.whiteToMove else "WHITE WINS"
                pygame.display.set_caption(winner)
                stopGame = True
            elif gs.itsDraw():  # Stalemate, Threefold Repetition
                pygame.display.set_caption("DRAW")
                stopGame = True
            elif gs.whiteToMove and not stopGame:
                PLAYERS[playerWhite](gs)
            elif not gs.whiteToMove and not stopGame:
                PLAYERS[playerBlack](gs, agent)
        drawGameState(screen, gs)
        clock.tick(MAX_FPS)
        pygame.display.flip()
    pygame.quit()

def humanPlay(gs):
    sqSelected = ()
    playerClicks = []
    col = row = 0
    moved = False
    while not moved:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    gs.goBackMove()
                elif event.key == pygame.K_RIGHT:
                    gs.goForthMove()
            if event.type == pygame.MOUSEBUTTONDOWN:
                location = pygame.mouse.get_pos()
                row = location[0] // SQ_SIZE
                col = location[1] // SQ_SIZE
                if sqSelected == (row, col):  # same square selected
                    sqSelected = ()  # deselect
                    playerClicks = []  # clearPlayerClicks
                else:
                    sqSelected = (row, col)
                    playerClicks.append(sqSelected)  # append 1st and 2nd clicks
                if len(playerClicks) == 2:
                    move = engine.Move(playerClicks[0], playerClicks[1], gs.board)
                    gs.makeMove(move)
                    #sqSelected = ()  # reset user clicks
                    #playerClicks = []
                    moved = True
        drawGameState(screen, gs)
        if len(playerClicks) == 1:
            color = "w" if gs.whiteToMove else "b"
            if gs.board[col][row][0] == color:
                highlightMoves(screen, gs, row, col)
        clock.tick(MAX_FPS)
        pygame.display.flip()

def randomPlay(gs):
    moves, movesID = gs.getValidMoves()
    move = moves[random.randint(0, len(moves) - 1)]
    gs.makeMove(move)

def minimaxPlay(gs):
    move = minmax.minimaxRoot(3, gs, True)
    gs.makeMove(move)

def agentPlay(gs, agent):
    state = gs.boardAsNumbers()
    action = agent.select_action(state)
    moves, movesID = gs.getValidMoves()
    gs.makeMove(moves[action])

PLAYERS = {"human": humanPlay, "random": randomPlay, "minimax": minimaxPlay, "alpha": agentPlay}

def highlightMoves(screen, gs, row, col):
    moves, movesID = gs.possiblePieceMoves(col, row)
    s = pygame.Surface((SQ_SIZE, SQ_SIZE))
    s.set_alpha(100)  # transparency value -> 0 transparent, 255 opaque
    s.fill(pygame.Color("blue"))
    screen.blit(s, (row * SQ_SIZE, col * SQ_SIZE))
    s.fill(pygame.Color("yellow"))
    for move in moves:
        if move.startCol == row and move.startRow == col:
            screen.blit(s, (move.endCol * SQ_SIZE, move.endRow * SQ_SIZE))

def drawGameState(screen, gs):
    drawBoard(screen)
    #add piece move highlight
    drawPieces(screen, gs.board)

def drawBoard(screen):
    colors = [pygame.Color("white"), pygame.Color("gray")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[(r+c) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def drawPieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != "--":
                screen.blit(IMAGES[piece], pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

if __name__ == "__main__":
    main()

