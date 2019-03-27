from lm_train import *
from log_prob import *
from preprocess import *
from math import log
from collections import defaultdict
import os

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
	"""
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)

	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model

	OUTPUT:
	AM :			(dictionary) alignment model structure

	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.

			LM['house']['maison'] = 0.5
	"""
	AM = {}
	SPACE = " "
    
	# Read training data
	eng, fre = read_hansard(train_dir, num_sentences)

	# Initializa AM
	AM = initialize(eng, fre)

	# E-M Algorithm
	for iter in range(max_iter):
		AM = em_step(AM, eng, fre)

	#Save Model
	with open(fn_AM+'.pickle', 'wb') as handle:
		pickle.dump(AM, handle, protocol = pickle.HIGHEST_PROTOCOL)

	return AM
    
# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
	"""
	Read up to num_sentences from train_dir.

	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider


	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.

	Make sure to read the files in an aligned manner.
	"""
	# TODO
	# Read training data
	E_EXT = ".e"
	F_EXT = ".f"
	E_LANG = "e"
	F_LANG = "f"
	SPACE = " "

	training_files = set()
	for path, subdirs, files in os.walk(train_dir):
		for name in files:
			training_files.add(os.path.splitext(os.path.join(path, name))[0])
	
	count = 0
	eng, fre = ([], [])
	for file in training_files:
		if (count == num_sentences):
			break

		e_file = open(file + E_EXT, "r")
		f_file = open(file + F_EXT, "r")

		for e_line, f_line in zip(e_file, f_file):
			if (count == num_sentences):
				break
			
			# range(1, -1) taken to ignore SENTSTART and SENTEND
			e_tokens = preprocess(e_line.strip(), E_LANG).split(SPACE)[1:-1]
			f_tokens = preprocess(f_line.strip(), F_LANG).split(SPACE)[1:-1]
			eng.append(e_tokens)
			fre.append(f_tokens)
			
			count += 1
	
	return (eng, fre)

def initialize(eng, fre):
	"""
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
	# TODO
	AM = {}
	AM['SENTSTART'] = {}
	AM['SENTSTART']['SENTSTART'] = 1
	AM['SENTEND'] = {}
	AM['SENTEND']['SENTEND'] = 1

	# Adding valid keys to AM
	for e_tokens, f_tokens in zip(eng, fre):
		for e_token in e_tokens:
			if (e_token not in AM.keys()):
				AM[e_token] = {}
			for f_token in f_tokens:
				AM[e_token][f_token] = 0.0

	# Step 1: Initialization with uniform distribution over possible words
	for e_token in AM:
		size = len(AM[e_token].keys())
		for f_token in AM[e_token]:
			AM[e_token][f_token] = 1.0/size

	return AM
    
def em_step(t, eng, fre):
	"""
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
	# TODO
	tcount = defaultdict(float)
	total = defaultdict(float)

	# E-Step
	for e_tokens, f_tokens in zip(eng, fre):
		for f_token in set(f_tokens):
			denom_c = 0.0
			for e_token in set(e_tokens):
				denom_c += (t[e_token][f_token] * f_tokens.count(f_token))

			for e_token in set(e_tokens):
				tcount[(f_token, e_token)] += (t[e_token][f_token] * f_tokens.count(f_token) * e_tokens.count(e_token)) / denom_c
				total[e_token] += (t[e_token][f_token] * f_tokens.count(f_token) * e_tokens.count(e_token)) / denom_c
	
	# M-Step
	for e_token in set(total):
		for f_token in t[e_token].keys():
			t[e_token][f_token] = tcount[(f_token, e_token)] / total[e_token]
	
	return t
