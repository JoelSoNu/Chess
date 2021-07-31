#!/bin/env python3
import pygame
import time
import sys
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img

if __name__ == "__main__":
    # Resize board game
    '''img = Image.open("chess_board.png")
    resized_img = img.resize((680, 580))
    resized_img.save("bigger_board.png")'''

    pygame.init()
    screen = pygame.display.set_mode((700, 600))
    pygame.display.set_caption("Chess board")
    board = pygame.image.load("bigger_board.png").convert()
    screen.blit(board, (10, 10))
    pygame.display.flip()
    running = True
    while(running):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()