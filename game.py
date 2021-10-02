#!/bin/env python3

import pygame
import time
import sys
import array
import math

from PIL import Image
import matplotlib.image as img

import engine

WIDTH = HEIGHT = 512
DIMENSION = 8  #dimension of chess board is 8x8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 30
IMAGES = {}


def loadImages():
    pieces = ['wp', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bp', 'bN', 'bB', 'bR', 'bQ', 'bK']
    for piece in pieces:
        IMAGES[piece] = pygame.transform.scale(pygame.image.load("images/" + piece + ".png"), (SQ_SIZE, SQ_SIZE))
    #we can acces an image by saying 'IMAGES['wp']


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    screen.fill(pygame.Color('white'))
    gs = engine.GameState()
    loadImages()
    pygame.display.set_caption("Chess board")
    running = True
    sqSelected = ()
    playerClicks = []
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                location = pygame.mouse.get_pos()
                row = location[0]//SQ_SIZE
                col = location[1]//SQ_SIZE
                if sqSelected == (row, col): #same square selected
                    sqSelected = () #deselect
                    playerClicks = [] #clearPlayerClicks
                else:
                    sqSelected = (row, col)
                    playerClicks.append(sqSelected) #append 1st and 2nd clicks
                if len(playerClicks) == 2:
                    move = engine.Move(playerClicks[0], playerClicks[1], gs.board)
                    gs.makeMove(move)
                    sqSelected = () #reset user clicks
                    playerClicks = []
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    gs.goBackMove()
                elif event.key == pygame.K_RIGHT:
                    gs.goForthMove()

        drawGameState(screen, gs)
        if len(playerClicks) == 1:
            color = "w" if gs.whiteToMove else "b"
            if gs.board[col][row][0] == color:
                highlightMoves(screen, gs, row, col)
        clock.tick(MAX_FPS)
        pygame.display.flip()
    pygame.quit()

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

