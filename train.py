import os 
import chess.pgn
from state import State

def getDataset():
	counter = 0
	# pgn files in the data folder
	X = []
	y = []
	for fn in os.listdir("data"):
		pgn = open(os.path.join("data", fn))
		while 1:
			try:
				game = chess.pgn.read_game(pgn)
			except Exception:
				break
			print("parsing game %d, examples %d " % (counter, len(X)))
			counter += 1
			value = {"1/2-1/2":0, "0-1":-1, "1-0":1}[game.headers["Result"]]
			board = game.board()
			for i, move in enumerate(game.mainline_moves()):
				board.push(move)
				ser = State(board).serialize()[:, :, 0]
				X.append(ser)
				y.append(value)
				#print(value, ser)
		exit(0)
		break


if __name__ == '__main__':
	getDataset()