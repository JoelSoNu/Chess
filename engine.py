#!/bin/env python3

class GameState():
    def __init__(self):
        self.board = [
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bp", "bp", "bp", "bp", "bp", "bp", "bp", "bp",],
            ["--", "--", "--", "--", "--", "--", "--", "--",],
            ["--", "--", "--", "--", "--", "--", "--", "--", ],
            ["--", "--", "--", "--", "--", "--", "--", "--", ],
            ["--", "--", "--", "--", "--", "--", "--", "--", ],
            ["wp", "wp", "wp", "wp", "wp", "wp", "wp", "wp", ],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]]
        self.whiteToMove = True
        self.moveLog = []
        self.undoneMoves = []

    #bugs
    def makeMove(self, move):
        self.board[move.startRow][move.startCol] = "--"
        self.board[move.endRow][move.endCol] = move.pieceMoved
        self.moveLog.append(move) #append the move so it can be undone later
        self.whiteToMove = not self.whiteToMove #swap players

    def goBackMove(self):
        if len(self.moveLog) > 0:
            move = self.moveLog.pop()
            self.undoneMoves.append(move)
            self.board[move.startRow][move.startCol] = move.pieceMoved
            self.board[move.endRow][move.endCol] = move.pieceCaptured
            self.whiteToMove = not self.whiteToMove  # swap players

    def goForthMove(self):
        if len(self.undoneMoves) > 0:
            move = self.undoneMoves.pop()
            self.moveLog.append(move)
            self.makeMove(move)

class Move():
    rowNotation = {"1": 7, "2": 6, "3": 5, "4": 4,
                   "5": 3, "6": 2, "7": 1, "8": 0}
    rowsToNotation = {v: k for k, v in rowNotation.items()}
    colNotation = {"a": 0, "b": 1, "c": 2, "d": 3,
                   "e": 4, "f": 5, "g": 6, "h": 7}
    colsToNotation = {v: k for k, v in colNotation.items()}

    def __init__(self, startSq, endSq, board):
        self.startRow = startSq[1]
        self.startCol = startSq[0]
        self.endRow = endSq[1]
        self.endCol = endSq[0]
        self.pieceMoved = board[self.startRow][self.startCol]
        self.pieceCaptured = board[self.endRow][self.endCol]

    def getChessNotation(self):
        return self.getSquare(self.startRow, self.startCol) + self.getSquare(self.endRow, self.endCol)

    def getSquare(self, r, c):
        return self.colsToNotation[c] + self.rowsToNotation[r]