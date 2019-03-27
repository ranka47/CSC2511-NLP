from preprocess import *
from lm_train import *
from math import log

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
	"""
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing

	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary

	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""

	#TODO: Implement by student.
	SPACE = " "
	prev_token = ""
	log_prob = 0
	for token in sentence.split(SPACE):
		if token == "SENTSTART":
			prev_token = token
			continue

		N_r = 0.0
		D_r = 0.0

		if smoothing:
			N_r = N_r + delta
			D_r = D_r + delta*vocabSize
		
		if prev_token in LM['bi'].keys() and token in LM['bi'][prev_token].keys():
			N_r = N_r + LM['bi'][prev_token][token]

		if prev_token in LM['uni'].keys():
			D_r = D_r + LM['uni'][prev_token]

		if N_r == 0.0:
			return float('-inf')
		else:
			log_prob = log_prob + log(float(N_r)/float(D_r), 2)
		
		prev_token = token
            
	return log_prob