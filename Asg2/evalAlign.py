#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import _pickle as pickle

import decode
from align_ibm1 import *
from BLEU_score import *
from lm_train import *

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'


discussion = """
Discussion :

{Enter your intriguing discussion (explaining observations/results) here}

"""

##### HELPER FUNCTIONS ########
def _getLM(data_dir, language, fn_LM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    use_cached  : (boolean) optionally load cached LM using pickle.

    Returns
    -------
    A language model 
    """
    if use_cached:
        return pickle.load(open(fn_LM, "rb"))
    else:
        filename, extension = os.path.splitext(fn_LM)
        return lm_train(data_dir, language, filename)

def _getAM(data_dir, num_sent, max_iter, fn_AM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data 
    num_sent    : (int) the maximum number of training sentences to consider
    max_iter    : (int) the maximum number of iterations of the EM algorithm
    fn_AM       : (string) the location to save the alignment model
    use_cached  : (boolean) optionally load cached AM using pickle.

    Returns
    -------
    An alignment model 
    """
    if use_cached:
        return pickle.load(open(fn_AM, "rb"))
    else:
        return align_ibm1(data_dir, num_sent, max_iter, fn_AM)

def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
    Parameters
    ----------
    eng_decoded : an array of decoded sentences
    eng         : an array of reference handsard
    google_refs : an array of reference google translated sentences
    n           : the 'n' in the n-gram model being used

    Returns
    -------
    An array of evaluation (BLEU) scores for the sentences
    """

    SPACE = " "
    bleu_scores = []
    for candidate, ref_eng, ref_google in zip(eng_decoded, eng, google_refs):
        candidate_length = len(candidate.split(SPACE))
        ref_eng_length = len(ref_eng.split(SPACE))
        ref_google_length = len(ref_google.split(SPACE))
    
        r_i = 0.0
        if (abs(candidate_length - ref_eng_length) < abs(candidate_length - ref_google_length)):
            r_i = ref_eng_length
        else:
            r_i = ref_google_length 

        brevity = r_i/candidate_length
        if brevity > 1:
            brevity = math.exp(1 - brevity)
        else:
            brevity = 1

        bleu_score = brevity
        for i in range(1, n+1):
            bleu_score *= math.pow(BLEU_score(candidate, [ref_eng, ref_google], i), 1.0/n)
        
        bleu_scores.append(bleu_score)
    
    return bleu_scores
        
   

def main(args):
    """
    #TODO: Perform outlined tasks in assignment, like loading alignment
    models, computing BLEU scores etc.

    (You may use the helper functions)

    It's entirely upto you how you want to write Task5.txt. This is just
    an (sparse) example.
    """
    root = "data/"
    results_dir = "results/"

    english_LM = None
    if os.path.exists(root + "Hansard/Training", "e", results_dir + "english.lm.pickle"):
        english_LM = _getLM(root + "Hansard/Training", "e", results_dir + "english.lm.pickle", use_cached=True)
    else:   
        english_LM = _getLM(root + "Hansard/Training", "e", results_dir + "english.lm", use_cached=False)

    AMs = []
    dataset_size = [1000, 10000, 15000, 30000]
    french_sents = []
    ref_english_sents = []
    ref_google_sents = []

    for french_sent, ref_english_sent, ref_google_sent in zip(open(root + "Hansard/Testing/Task5.f", "r"), open(root + "Hansard/Testing/Task5.e"), open(root + "Hansard/Testing/Task5.google.e")):
        french_sents.append(preprocess(french_sent.strip(), 'f'))
        ref_english_sents.append(preprocess(ref_english_sent.strip(), 'e'))
        ref_google_sents.append(preprocess(ref_google_sent.strip(), 'e'))

    for num_sents in dataset_size:
        # if os.path.exists(results_dir + "AM_" + str(num_sents) + ".pickle"):
        #     AM = _getAM(root + "Hansard/Training", num_sents, 1000, results_dir + "AM_" + str(num_sents) + ".pickle", use_cached=True)
        # else:   
        #     AM = _getAM(root + "Hansard/Training", num_sents, 1000, results_dir + "AM_" + str(num_sents), use_cached=False)
        
        AM = _getAM(root + "Hansard/Training", num_sents, 50, results_dir + "AM_" + str(num_sents), use_cached=False)
        AMs.append(AM)
        print("############### Trained on ", num_sents, " sentences")
        for n_gram in range(1, 4):
            print("N-gram: ", n_gram)
            decoded_sents = []

            for french_sent, ref_english_sent, ref_google_sent in zip(french_sents, ref_english_sents, ref_google_sents):
                decoded_sent = decode.decode(french_sent, english_LM, AM)
                decoded_sents.append(decoded_sent)

            bleu_score = _get_BLEU_scores(decoded_sents, ref_english_sents, ref_google_sents, n_gram)
            print(" ".join([str(score) for score in bleu_score]))


    ## Write Results to Task5.txt (See e.g. Task5_eg.txt for ideation). ##

    f = open("Task5.txt", 'w+')
    f.write(discussion) 
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")

    for i, AM in enumerate(AMs):
        
        f.write("""\n### Evaluating AM model: %s sentences ### \n""" % (str(dataset_size[i])))
        # Decode using AM #
        # Eval using 3 N-gram models #
        all_evals = []
        for n in range(1, 4):
            f.write("""\nBLEU scores with N-gram (n) = %s: """ % (str(n)))
            decoded_sents = []
            
            for french_sent, ref_english_sent, ref_google_sent in zip(french_sents, ref_english_sents, ref_google_sents):
                decoded_sent = decode.decode(french_sent, english_LM, AM)
                decoded_sents.append(decoded_sent)

            evals = _get_BLEU_scores(decoded_sents, ref_english_sents, ref_google_sents, n)
            for v in evals:
                f.write("\t%1.4f" % (v))
            all_evals.append(evals)

        f.write("\n\n")

    f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use parser for debugging if needed")
    args = parser.parse_args()

    main(args)