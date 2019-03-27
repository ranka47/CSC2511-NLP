import math

def calc_brevity(candidate, references):
    SPACE = " "
    candidate_length = len(candidate.split(SPACE))
    nearest_diff = math.inf
    r_i = 0.0

    for reference in references:
        ref_length = len(reference.split(SPACE))
        if (abs(candidate_length - ref_length) < nearest_diff):
            r_i = ref_length
    
    brevity = r_i/candidate_length

    if brevity < 1:
        return 1
    else:
        return math.exp(1 - brevity)

def BLEU_score(candidate, references, n, brevity=False):
    """
    Calculate the BLEU score given a candidate sentence (string) and a list of reference sentences (list of strings). n specifies the level to calculate.
    n=1 unigram
    n=2 bigram
    ... and so on
    
    DO NOT concatenate the measurments. N=2 means only bigram. Do not average/incorporate the uni-gram scores.
    
    INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.

	
	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""
	
	#TODO: Implement by student.
    SPACE = " "
    candidate_list = candidate.split(SPACE)

    N_r = 0.0
    D_r = len(candidate_list) - n + 1
    for i in range(n, len(candidate_list)):
        find_string = SPACE.join(candidate_list[i-n: i])
        for reference in references:
            if find_string in reference:
                N_r += 1
                break

    if brevity:
        bleu_score = calc_brevity(candidate, references) * N_r/D_r
    else:
        bleu_score = N_r/D_r

    return bleu_score