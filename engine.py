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
        self.moveFunctions = {'p': self.pawnMoves, 'N': self.knightMoves, 'B': self.bishopMoves,
                              'R': self.rockMoves, 'Q': self.queenMoves, 'K': self.kingMoves}

    #bugs
    def makeMove(self, move):
        moves, movesID = self.possibleMoves()
        print(movesID)
        if move.moveID in movesID:
            print(move.getChessNotation())
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

    def possibleMoves(self):
        moves = []
        movesID = []
        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
                player = self.board[r][c][0]
                if(player == "w" and self.whiteToMove) or (player == "b" and not self.whiteToMove):
                    piece = self.board[r][c][1]
                    self.moveFunctions[piece](r, c, moves, movesID)
        return moves, movesID

    def pawnMoves(self, r, c, moves, movesID):
        if self.whiteToMove:
            if self.board[r-1][c] == "--":
                move = Move((c, r), (c, r-1), self.board)
                moves.append(move)
                movesID.append(move.moveID)
                print(move.moveID)
                if r == 6 and self.board[r-2][c] == "--":
                    move = Move((c, r), (c, r-2), self.board)
                    moves.append(move)
                    movesID.append(move.moveID)
        else:
            pass

    def knightMoves(self, r, c, moves, movesID):
        pass

    def bishopMoves(self, r, c, moves, movesID):
        pass

    def rockMoves(self, r, c, moves, movesID):
        pass

    def queenMoves(self, r, c, moves, movesID):
        pass

    def kingMoves(self, r, c, moves, movesID):
        pass

    def notIllegalMove(self, move):
        if move.pieceCaptured[1] == "K":
            return False
        #boardCopy = self.board
        #boardCopy[move.startRow][move.startCol] = "--"
        #boardCopy[move.endRow][move.endCol] = move.pieceMoved
        #moves, movesID = self.possibleMoves(boardCopy)
        #for move in moves:
         #   if move.pieceCaptured[1] == "K":
          #      return False
        return True

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
        self.moveID = self.getChessNotation()

    def getChessNotation(self):
        return self.getSquare(self.startRow, self.startCol) + self.getSquare(self.endRow, self.endCol)

    def getSquare(self, r, c):
        return self.colsToNotation[c] + self.rowsToNotation[r]