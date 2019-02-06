import sys
import argparse
import os
import json
import re
import string
import spacy
import html

indir = '/u/cs401/A1/data/';
required_files_dir = 'datasets/reddit/'
student_id = 0
EMPTY_STRING = ""
SPACE = " "
NEWLINE_REGEX = r'\n'
nlp = spacy.load('en', disable=['parser', 'ner'])

# Characters taken from https://perishablepress.com/stop-using-unsafe-characters-in-urls/ excluding round brackets
URL_REGEX = r'(https:\/\/|http:\/\/|www\.)[$.+!*,;%?:@=&\w\-\'\/]*'  
abbrev = []
stopwords = []

def detect_abbrev(pattern):
    if pattern.group(0).strip() in abbrev:
        return pattern.group(0)
    else:
        return pattern.group(1) + SPACE + "." + SPACE

def detect_stopwords(pattern):
    if pattern.group(1).strip().lower() in stopwords:
        return ""
    else:
        return pattern.group(0)

def detect_eos(line, is_tokenized):
    """
    No EOS assigned at the end of `line`
    """
    tagged_line = ""

    if (is_tokenized):
        word_line = re.sub(r"(\S+)\/(\S+)", r"\1", line)
        tags_list = re.sub(r"(\S+)\/(\S+)", r"\2", line).split(SPACE)
    
    if (len(word_line.split(" ")) != len(tags_list)):
        print("Error. Size not matching!!!!")
        import pdb; pdb.set_trace()

    # Newline after EOS punctuations
    word_line = re.sub(r"(\w) ([\.!\?;:\"]+) (\w)", r"\1 \2 \n \3", word_line)

    # Move EOS if punctuations followed by double quotations
    word_line = re.sub(r" (\.|!|\?) \" ", r" \1 \" \n ", word_line)

    # Disqualify boundary with ? or ! if followed by a lowercase letter
    word_line = re.sub(r"(\?|!|\.) \n([a-z])", r"\1 \2", word_line)

    index = 0
    for word in word_line.split(SPACE):
        if (word == "\n"):
            tagged_line = tagged_line + "\n/\n "
        else:
            tagged_line = tagged_line + word + "/" + tags_list[index] + SPACE
            index = index + 1
    return tagged_line.strip()
 
def tagger(line):
    if ('/' in line):
        print("Backslash detected in line =>", line)

    tokenize_line = line.split(SPACE)
    tagged_line = ""
    doc = spacy.tokens.Doc(nlp.vocab, words=tokenize_line)
    doc = nlp.tagger(doc)
    for token in doc:
        tagged_line = tagged_line + token.text + "/" + token.tag_ + SPACE
    return tagged_line

def lemmatize(line, is_tokenized):
    if (is_tokenized):
        line = re.sub(r"(\S+)\/(\S+)", r"\1", line)

    tokenize_line = line.split(SPACE)
    lemmatize_line = ""
    doc = spacy.tokens.Doc(nlp.vocab, words=tokenize_line)
    doc = nlp.tagger(doc)
    for token in doc:
        if (token.lemma_.startswith("-")):
            lemmatize_line = lemmatize_line + token.text + "/" + token.tag_ + SPACE
        else:
            lemmatize_line = lemmatize_line + token.lemma_ + "/" + token.tag_ + SPACE

    return lemmatize_line

def preproc1(comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    if (comment == "[deleted]"):
        return EMPTY_STRING

    modComm = comment
    if 1 in steps and not re.fullmatch(r"\s*", modComm):
        modComm = re.sub(NEWLINE_REGEX, SPACE, modComm)
    if 2 in steps and not re.fullmatch(r"\s*", modComm):
        modComm = modComm.strip()
        modComm = html.unescape(modComm)
    if 3 in steps and not re.fullmatch(r"\s*", modComm):
        modComm = modComm.strip()
        modComm = re.sub(URL_REGEX, EMPTY_STRING, modComm)
    if 4 in steps and not re.fullmatch(r"\s*", modComm):
        modComm = modComm.strip()
        # Replace ." ?" !" as . " ? " ! "
        modComm = re.sub(r"(\w)(!|\.|\?)[\s]*(\")", r"\1 \2 \3", modComm)
        
        # Replace ", as " ,
        modComm = re.sub(r"\", ", "\" , ", modComm)

        # Tokenize grammatically incorrect multiple punctuations as a single token  (breaks grammatically incorrect <dogs'.>)
        modComm = re.sub(r"(([^\s\w]{2,})(\")|([^\s\w]{2,}))", r" \2\4 \3 ", modComm)

        # Tokenize other punctuations :;[]{}()<>!?, coming after a character
        modComm = re.sub(r"(\w)([:;<>?!\"\[\]\{\}\(\)]){1}", r"\1 \2 ", modComm)

        # Tokenize other punctuations :;<>,?!"[]{}() coming before a character
        modComm = re.sub(r"([:;<>?!\"\[\]\{\}\(\)]){1}(\w)", r" \1 \2", modComm)

        # Tokenize comma coming before a character (comma handled separately to prevent from splitting numbers, eg: 10,000)
        modComm = re.sub(r",([a-zA-Z_])", r", \1", modComm) 

        # Tokenize comma coming after a character
        modComm = re.sub(r"([a-zA-Z_]),", r"\1 ,", modComm) 

        # Tokenize period ignoring abbreviations
        modComm = re.sub(r"(\w[\w\.]*\w)[\.]( |$)", detect_abbrev, modComm)

    if 5 in steps and not re.fullmatch(r"\s*", modComm):
        modComm = modComm.strip()
        modComm = re.sub(r"(\w)(n't|'d|'ve|'m|'re|'ll|'n)", r"\1 \2 ", modComm)
        modComm = re.sub(r"s'( |$)", r"s '\1", modComm)
    if 6 in steps and not re.fullmatch(r"\s*", modComm):
        modComm = modComm.strip()
        modComm = re.sub(r"\s+", SPACE, modComm)
        modComm = tagger(modComm)
    if 7 in steps and not re.fullmatch(r"\s*", modComm):
        modComm = modComm.strip()

        # Case insensitive in searching for stopwords
        modComm = re.sub(r"(\S+)\/(\S+)", detect_stopwords, modComm)
    if 8 in steps and not re.fullmatch(r"\s*", modComm):
        modComm = modComm.strip()
        modComm = re.sub(r"\s+", SPACE, modComm)
        modComm = lemmatize(modComm, 6 in steps)
    if 9 in steps and not re.fullmatch(r"\s*", modComm):
        modComm = modComm.strip()
        modComm = re.sub(r"\s+", SPACE, modComm)
        modComm = detect_eos(modComm, 6 in steps or 8 in steps)
    if 10 in steps and not re.fullmatch(r"\s*", modComm):
        modComm = modComm.strip()
        modComm = re.sub(r"(\S+)\/(\S+)", lambda pattern: pattern.group(1).lower() + "/" + pattern.group(2), modComm)
    return modComm

def preproc(data, category):
    for i in range(len(data)):
        json_line = json.loads(data[i])

        print("####################Original String####################")
        print (json_line['body'])

        preproc_body = preproc1(json_line['body'])

        print("####################Parsed String####################")
        print(preproc_body.encode('unicode_escape').decode('utf-8'))
    
        json_line['body'] = preproc_body
        json_line['cat'] = category

        data[i] = json.JSONEncoder().encode(json_line)
    return data

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for filename in files:
            fullFile = os.path.join(subdir, filename)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))
            
            start_index = student_id % len(data)
            if (len(data) - (start_index + 1) < args.max):
                data = data[start_index : ] + data[0: args.max - len(data) + start_index]
            else:
                data = data[start_index : start_index + args.max]
            
            preproc_data = preproc(data, filename)
            allOutput = allOutput + preproc_data

            # TODO: select appropriate args.max lines
            # TODO: read those lines with something like `j = json.loads(line)`
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'filename' (e.g., 'Alt', 'Right', ...) 
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='s')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000, type=int)
    parser.add_argument("-l", "--local", help="Specify the flag when running locally", action="store_true")
    args = parser.parse_args()

    student_id = args.ID[0]
    abbrev = open(required_files_dir + "wordlists/abbrev.english").read().splitlines()
    stopwords = open(required_files_dir + "wordlists/StopWords").read().splitlines()

    # Changes input data directory when running in a local system
    if (args.local):
        indir = "./datasets/reddit/data/"

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
        
    main(args)
