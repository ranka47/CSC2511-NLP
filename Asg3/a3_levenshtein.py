import os
import numpy as np

# dataDir = '/u/cs401/A3/data/'
dataDir = 'data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> Levenshtein("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> Levenshtein("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> Levenshtein("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    UP = 1
    LEFT = 2
    UP_LEFT = 3

    n = len(r)
    m = len(h)
    R = np.zeros((n+1, m+1))    # Matrix Distances
    B = np.zeros((n+1, m+1))    # Backtracking Matrix
    
    R[0, :] = np.arange(R[0, :].shape[0])
    R[:, 0] = np.arange(R[:, 0].shape[0])
    B[0, :] = LEFT
    B[:, 0] = UP
    # Partial Alignment Errors being calculated and stored in R

    for i in range(1, n+1):
        for j in range(1, m+1):
            deletes = R[i-1, j] + 1
            substitutes = R[i-1, j-1] + (0 if r[i-1] == h[j-1] else 1)
            inserts = R[i, j-1] + 1
            R[i, j] = min([deletes, substitutes, inserts])

            if (R[i, j] == deletes):
                B[i, j] = UP
            elif (R[i, j] == inserts):
                B[i, j] = LEFT
            else:
                B[i, j] = UP_LEFT
    
    # Backtracking and calculating WER, nS, nI, nD
    i = n
    j = m
    deletes = inserts = substitutes = 0
    while(not(i == 0 and j == 0)):
        if (B[i, j] == UP):
            deletes += 1; i = i - 1
        elif (B[i, j] == LEFT):
            inserts += 1; j = j - 1
        elif (B[i, j] == UP_LEFT):
            substitutes += (0 if r[i-1] == h[j-1] else 1)
            i = i - 1; j = j - 1; 
        else:
            print("Unexpected condition faced!", B[i, j])
    
    if (n == 0):
        wer = np.inf
    else:
        wer = (substitutes + inserts + deletes)/n

    return wer, substitutes, inserts, deletes

def preprocess(string):
    import re
    string = re.sub(r"[^a-zA-Z0-9\s\[\]]", r"", string)
    string = string.lower()
    string = string.strip().split()
    # import pdb; pdb.set_trace() 
    return string[2:]

if __name__ == "__main__":
    # print( 'TODO' ) 
    GOOGLE_FILENAME = "transcripts.Google.txt"
    KALDI_FILENAME = "transcripts.Kaldi.txt"
    HUMAN_FILENAME = "transcripts.txt"

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            google_file = open(os.path.join( dataDir, speaker, GOOGLE_FILENAME), "r")
            kaldi_file = open(os.path.join( dataDir, speaker, KALDI_FILENAME), "r")
            human_file = open(os.path.join( dataDir, speaker, HUMAN_FILENAME), "r")

            count = 0
            for kaldi_hypo, google_hypo, ref in zip(kaldi_file, google_file, human_file):
                kaldi_hypo, google_hypo, ref = preprocess(kaldi_hypo), preprocess(google_hypo), preprocess(ref)

                wer, s, i, d = Levenshtein(ref, kaldi_hypo)
                print("%5s %6s %2d %4.3f S:%4d, I:%4d, D:%4d" % (speaker, "Kaldi", count, wer, s, i, d))
                wer, s, i, d = Levenshtein(ref, google_hypo)
                print("%5s %6s %2d %4.3f S:%4d, I:%4d, D:%4d" % (speaker, "Google", count, wer, s, i, d))

                count = count + 1
                
