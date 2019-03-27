from preprocess import *
import pickle
import os
import glob

def lm_train(data_dir, language, fn_LM):
	"""
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM

	INPUTS:

	data_dir	: (string) The top-level directory continaing the data from which
					to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained

	OUTPUT

	LM			: (dictionary) a specialized language model

	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which 
	incorporate unigram or bigram counts

	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
			LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
	"""

	# TODO: Implement Function
	# Initialization of the n-gram models
	language_model = {'uni': {}, 'bi': {}}

	# Useful GLOBAL variables
	SPACE = " "
	
	for file in glob.glob(data_dir + "/**/*." + language, recursive=True):
		with open(file, 'r') as input_file:
			for line in input_file.readlines():
				preproc_string = preprocess(line.strip(), language)

				# Initialized as empty string for SENTSTART
				prev_token = ""
				for token in preproc_string.split(SPACE):

					# Initialize for previous token in the bi-gram model
					if prev_token not in language_model['bi']:
						language_model['bi'][prev_token] = {}

					# Update count for uni-gram model
					if token in language_model['uni'].keys():
						language_model['uni'][token] += 1
					else:
						language_model['uni'][token] = 1

					# Update count for bi-gram model with respect to the previous token					
					if token in language_model['bi'][prev_token].keys():
						language_model['bi'][prev_token][token] += 1
					else:
						language_model['bi'][prev_token][token] = 1
					
					# Update the previous token
					prev_token = token

	#Save Model
	with open(fn_LM+'.pickle', 'wb') as handle:
		pickle.dump(language_model, handle, protocol = pickle.HIGHEST_PROTOCOL)

	return language_model