import engine
import random

def minimaxRoot(depth, gs, isMaximizing):
    moves, movesID = gs.getValidMoves()
    bestMove = -9999
    bestMoveFinal = None
    for move in moves:
        gs.movePiece(move)
        value = max(bestMove, minimax(depth - 1, gs, -10000, 10000, not isMaximizing))
        gs.undoMove()
        if( value > bestMove):
            print("Best score: ", str(bestMove))
            bestMove = value
            bestMoveFinal = move
    return bestMoveFinal


def minimax(depth, gs, alpha, beta, is_maximizing):
    moves, movesID = gs.getValidMoves()
    if(depth == 0):
        return -evaluation(gs)
    if(is_maximizing):
        bestMove = -9999
        for move in moves:
            gs.movePiece(move)
            bestMove = max(bestMove,minimax(depth - 1, gs, alpha, beta, not is_maximizing))
            gs.undoMove()
            alpha = max(alpha, bestMove)
            if beta <= alpha:
                return bestMove
        return bestMove
    else:
        bestMove = 9999
        for move in moves:
            gs.movePiece(move)
            bestMove = min(bestMove, minimax(depth - 1, gs, alpha, beta, not is_maximizing))
            gs.undoMove()
            beta = min(beta, bestMove)
            if(beta <= alpha):
                return bestMove
        return bestMove

'''
def calculateMove(board):
    possible_moves = board.legal_moves
    if(len(possible_moves) == 0):
        print("No more possible moves...Game Over")
        sys.exit()
    bestMove = None
    bestValue = -9999
    n = 0
    for x in possible_moves:
        move = chess.Move.from_uci(str(x))
        board.push(move)
        boardValue = -evaluation(board)
        board.pop()
        if(boardValue > bestValue):
            bestValue = boardValue
            bestMove = move

    return bestMove
'''

def evaluation(gs):
    board = gs.g()
    evaluation = 0
    imWhite = gs.whiteToMove
    color = "w" if imWhite else "b"
    for i in range(0, len(board)):
        evaluation = evaluation + getPieceValue(board[i][1]) if board[i][0] == color else evaluation - getPieceValue(board[i][1])
    evaluation = evaluation + random.uniform(0, 1)
    #print(evaluation)
    # Need more heuristics so maybe all moves have same value and do first one repeating
    # it if there is no move that wins material
    return evaluation


def getPieceValue(piece):
    if piece == "-":
        return 0
    value = 0
    if piece == "P" or piece == "p":
        value = 10
    if piece == "N" or piece == "n":
        value = 30
    if piece == "B" or piece == "b":
        value = 30
    if piece == "R" or piece == "r":
        value = 50
    if piece == "Q" or piece == "q":
        value = 90
    if piece == 'K' or piece == 'k':
        value = 900
    return value
