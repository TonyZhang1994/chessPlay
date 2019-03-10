import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class chessData(Dataset):
	def __init__(self):
		data = np.load("trainData/trainData1k.npz")
		self.X = data['X']
		self.y = data['y']
		print(self.X.shape, self.y.shape)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, index):
		return {"X": self.X[index], "y": self.y[index]}

chessData = chessData()
