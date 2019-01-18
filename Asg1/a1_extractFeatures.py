import numpy as np
import sys
import argparse
import os
import json
import re
import csv

featdir = '/u/cs401/A1/feats/';
wordlistsdir = '/u/cs401/A1/wordlists/';

CAT_TO_INT = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}
BGL_FEATURES = {}
WARRINGER_FEATURES = {}
LIWC_FEATURES = {}
SPACE = " "
slangs = []

# whitespace is added to identify the breaks
first_person_regex = re.compile(r"(^| )(i|me|my|mine|we|us|our|ours)\/")
second_person_regex = re.compile(r"(^| )(you|your|yours|u|ur|urs)\/")
third_person_regex = re.compile(r"(^| )(he|him|his|she|her|hers|it|its|they|them|their|theirs)\/")
coord_conjunc_regex = re.compile(r"\/CC( |$)")
past_tense_regex = re.compile(r"\/VBD( |$)")
future_tense_regex = re.compile(r"(^| )(will|'ll|gonna)\/")
going_to_verb_regex = re.compile(r"(^| )going\/\S+ to\/\S+ \S+\/VB( |$)")
comma_regex = re.compile(r"(^| ),\/,( |$)")
multiple_punctuation_regex = re.compile(r"(^| )(([^\s\w]{2,})(\")|([^\s\w]{2,}))\/")
common_noun_regex = re.compile(r"\/(NN|NNS)( |$)")
proper_noun_regex = re.compile(r"\/(NNP|NNPS)( |$)")
adverbs_regex = re.compile(r"\/(RB|RBR|RBS)( |$)")
wh_words_regex = re.compile(r"\/(WDT|WP|WP\$|WRB)( |$)")
uppercase_words_regex = re.compile("(^| )[A-Z]{3,}")
eos_regex = re.compile(r"\n( |$)")
word_regex = re.compile(r"\S\/\S")
word_with_atleast_one_character = re.compile(r"[^\s\w]*\w\S*\/")

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    
    feats = np.zeros((29,))
    feats[0] = len(first_person_regex.findall(comment))
    feats[1] = len(second_person_regex.findall(comment))
    feats[2] = len(third_person_regex.findall(comment))
    feats[3] = len(coord_conjunc_regex.findall(comment))
    feats[4] = len(past_tense_regex.findall(comment))
    feats[5] = len(future_tense_regex.findall(comment)) + len(going_to_verb_regex.findall(comment))
    feats[6] = len(comma_regex.findall(comment))
    feats[7] = len(multiple_punctuation_regex.findall(comment))
    feats[8] = len(common_noun_regex.findall(comment))
    feats[9] = len(proper_noun_regex.findall(comment))
    feats[10] = len(adverbs_regex.findall(comment))
    feats[11] = len(wh_words_regex.findall(comment))
    feats[12] = count_slangs(comment)
    feats[13] = len(uppercase_words_regex.findall(comment))
    feats[16] = len(eos_regex.findall(comment))
    feats[14] = 0 if (feats[16] == 0) else len(word_regex.findall(comment))/feats[16]  
    
    temp_count = 0
    for word in word_with_atleast_one_character.findall(comment):
        temp_count = temp_count + len(word) - 1
    
    feats[15] = 0 if ((feats[14] - feats[7]) == 0) else temp_count/(feats[14] - feats[7])
    feats[17:23] = extract_BGL(comment)
    feats[23:29] = extract_warringer(comment)

    return feats

def count_slangs(line):
    tokenize_line = re.sub(r"(\S+)\/(\S+)", r"\1", line).split(SPACE)
    count = 0
    for token in tokenize_line:
        if token in slangs:
            count = count + 1

    return count
    
def extract_BGL(line):
    aoa = img = fam = []
    tokens = re.sub(r"(\S+)\/(\S+)", r"\1", line).split(SPACE)
    zero_output = True

    for token in tokens:
        if token in BGL_FEATURES.keys():
            zero_output = False
            aoa.append(BGL_FEATURES[token]["AoA"])
            img.append(BGL_FEATURES[token]["IMG"])
            fam.append(BGL_FEATURES[token]["FAM"])
    
    if zero_output:
        return np.zeros((6,))

    aoa = np.array(aoa)
    img = np.array(img)
    fam = np.array(fam)

    return [aoa.mean(), img.mean(), fam.mean(), aoa.std(), img.std(), fam.std()] 

def extract_warringer(line):
    v_mean_sum = a_mean_sum = d_mean_sum = []
    tokens = re.sub(r"(\S+)\/(\S+)", r"\1", line).split(SPACE)
    zero_output = True

    for token in tokens:
        if token in WARRINGER_FEATURES.keys():
            zero_output = False
            v_mean_sum.append(WARRINGER_FEATURES[token]["V.Mean.Sum"])
            a_mean_sum.append(WARRINGER_FEATURES[token]["A.Mean.Sum"])
            d_mean_sum.append(WARRINGER_FEATURES[token]["D.Mean.Sum"])

    if zero_output:
        return np.zeros((6,))
    v_mean_sum = np.array(v_mean_sum)
    a_mean_sum = np.array(a_mean_sum)
    d_mean_sum = np.array(d_mean_sum)

    return [v_mean_sum.mean(), a_mean_sum.mean(), d_mean_sum.mean(), v_mean_sum.std(), a_mean_sum.std(), d_mean_sum.std()]

def extract_LIWC(cat, id):
    return LIWC_FEATURES[cat][id]

def extract(data):
    feats = np.zeros((173+1,))
    json_data = json.loads(data)
    data_body = json_data["body"]
    data_id = json_data["id"]
    data_cat = json_data["cat"]
    feats[0:29] = extract1(data_body)
    feats[29:-1] = extract_LIWC(data_cat, data_id)
    feats[-1] = CAT_TO_INT[data_cat]
    return feats

def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))
    slangs = open(wordlistsdir + "slangs.list").read().splitlines()
    global BGL_FEATURES, WARRINGER_FEATURES, LIWC_FEATURES
    BGL_FEATURES = {
        row["WORD"] : {
            "AoA": float(row["AoA (100-700)"]),
            "IMG": float(row["IMG"]),
            "FAM": float(row["FAM"])
            } 
            for row in csv.DictReader(open(os.path.join(wordlistsdir, "BristolNorms+GilhoolyLogie.csv")))
                if ((row["AoA (100-700)"] != "") or (row["IMG"] != "") or (row["FAM"] != ""))
        }
    print("BGL Features read")
    WARRINGER_FEATURES = {
        row["Word"] : {
            "V.Mean.Sum": float(row["V.Mean.Sum"]),
            "A.Mean.Sum": float(row["A.Mean.Sum"]),
            "D.Mean.Sum": float(row["D.Mean.Sum"])
            } 
            for row in csv.DictReader(open(os.path.join(wordlistsdir, "Ratings_Warriner_et_al.csv")))
                if ((row["V.Mean.Sum"] != "") or (row["A.Mean.Sum"] != "") or (row["D.Mean.Sum"] != ""))
        }
    print("Warringer Features read")
    LIWC_FEATURES = {
        cat : {
            id.strip(): row for id, row in zip(open(os.path.join(featdir, cat + "_IDs.txt")), np.load(os.path.join(featdir, cat + "_feats.dat.npy"))) 
        } 
        for cat in ["Alt", "Center", "Right", "Left"]
    }
    print("LIWC Features read")

    # TODO: your code here
    for i in range(len(data)):
        feats[i] = extract(data[i])
        import pdb; pdb.set_trace()

    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-l", "--local", help="Specify the flag when running locally", action="store_true")

    args = parser.parse_args()
                 
    # Changes input data directory when running in a local system
    if (args.local):
        featdir = "./datasets/reddit/feats/"
        wordlistsdir = "./datasets/reddit/wordlists/"

    main(args)

