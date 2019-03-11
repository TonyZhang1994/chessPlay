from keras.models import model_from_json
import os
from state import State
from keras.models import load_model
import numpy as np

class Evaluator(object):
	"""docstring for Evaluator"""
	def __init__(self):
		json_file = open('model.json', 'r')

		loaded_model_json = json_file.read()
		json_file.close()
		self.loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		self.loaded_model.load_weights("model.h5")
		print("Loaded model from disk")

		self.loaded_model.compile(loss='categorical_crossentropy', 
		 					optimizer='adam',
		 					metrics=['accuracy'])

	def __call__(self, s):
		return self.loaded_model.predict(s.serialize().reshape(1,8,8,5))

	
if __name__ == '__main__':
	eva = Evaluator()
	s = State()
	for i in s.edges():
		s.board.push(i)
		print(s.board)
		print(eva(s))
		s.board.pop()
		
